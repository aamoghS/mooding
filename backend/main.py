from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import webbrowser
import requests
import os
import base64
import numpy as np
import cv2
from cnn_utils import EmotionDetector

app = FastAPI()
load_dotenv()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize emotion detector
emotion_detector = EmotionDetector()

client_id = 'a7c99eebbf6843d383adf6ffa6b93854'
client_secret = 'a3bc41920291443787dc5f5a7cf5638d'
redirect_uri = 'http://127.0.0.1:8000/callback'

@app.get("/login")
def login():
    scope = "user-library-read playlist-read-private"
    auth_url = (
        "https://accounts.spotify.com/authorize"
        f"?response_type=code&client_id={client_id}"
        f"&redirect_uri={redirect_uri}&scope={scope}"
    )
    return RedirectResponse(url=auth_url)

@app.get("/callback")
async def callback(request: Request):
    code = request.query_params.get("code")

    if not code:
        raise HTTPException(status_code=400, detail="err")

    token = await get_access_token(code)
    profile = get_user_profile(token)
    playlists = get_user_playlists(token)
    
    # Create a URL with the data as query parameters
    frontend_url = f"http://localhost:3000/profile?profile={profile}&playlists={playlists}"
    return RedirectResponse(url=frontend_url)

async def get_access_token(code: str):
    token_url = "https://accounts.spotify.com/api/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code == 200:
        token_info = response.json()
        return token_info['access_token']
    else:
        raise HTTPException(status_code=response.status_code, detail="err")

def get_user_profile(access_token: str):
    url = "https://api.spotify.com/v1/me"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail="err")

def get_user_playlists(access_token: str):
    url = "https://api.spotify.com/v1/me/playlists"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": "err",
            "details": response.json()
        }

def refresh_access_token(refresh_token: str):
    token_url = "https://accounts.spotify.com/api/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise HTTPException(status_code=response.status_code, detail="err")

@app.post("/detect-emotion")
async def detect_emotion(request: Request):
    try:
        data = await request.json()
        image_data = data.get('image')
        
        if not image_data:
            raise HTTPException(status_code=400, detail="err")
        
        # Remove the data URL prefix if present
        if image_data.startswith('data:image/jpeg;base64,'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect emotions
        results = emotion_detector.detect_emotion(frame)
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="err")

@app.post("/find-matching-song")
async def find_matching_song(request: Request):
    try:
        data = await request.json()
        emotion = data.get('emotion')
        playlists = data.get('playlists')
        target_genres = data.get('genres', [])

        if not emotion or not playlists or not target_genres:
            raise HTTPException(status_code=400, detail="Missing required data")

        # Get a new access token using the stored profile data
        token_url = "https://accounts.spotify.com/api/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        auth_string = f"{client_id}:{client_secret}"
        auth_bytes = auth_string.encode('utf-8')
        auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')
        headers["Authorization"] = f"Basic {auth_base64}"

        data = {
            "grant_type": "client_credentials"
        }

        token_response = requests.post(token_url, headers=headers, data=data)
        
        if token_response.status_code != 200:
            raise HTTPException(status_code=token_response.status_code, detail="Failed to get access token")
        
        access_token = token_response.json()['access_token']

        # Now search through playlists with the valid access token
        for playlist in playlists.get('items', []):
            playlist_id = playlist['id']
            
            # Get playlist tracks
            tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
            headers = {
                "Authorization": f"Bearer {access_token}"
            }
            response = requests.get(tracks_url, headers=headers)
            
            if response.status_code == 200:
                tracks = response.json().get('items', [])
                
                # For each track, get its audio features and genres
                for track in tracks:
                    if not track.get('track'):  # Skip if track is None or empty
                        continue
                        
                    track_data = track['track']
                    if not track_data.get('artists'):  # Skip if no artists
                        continue
                        
                    artist_id = track_data['artists'][0]['id']
                    
                    # Get artist genres
                    artist_url = f"https://api.spotify.com/v1/artists/{artist_id}"
                    artist_response = requests.get(artist_url, headers=headers)
                    
                    if artist_response.status_code == 200:
                        artist_data = artist_response.json()
                        artist_genres = artist_data.get('genres', [])
                        
                        # Check if any of the artist's genres match our target genres
                        matching_genres = set(artist_genres) & set(target_genres)
                        
                        if matching_genres:
                            return {
                                "name": track_data['name'],
                                "artist": track_data['artists'][0]['name'],
                                "genres": list(matching_genres),
                                "preview_url": track_data.get('preview_url'),
                                "external_url": track_data['external_urls']['spotify']
                            }
        
        # If no matching song found
        return {"message": "No matching song found"}

    except Exception as e:
        print(f"Error in find_matching_song: {str(e)}")  # Add logging for debugging
        raise HTTPException(status_code=500, detail=str(e))