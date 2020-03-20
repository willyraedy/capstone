from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from dotenv import load_dotenv
import os
import numpy as np
import time

load_dotenv()

IBM_API_KEY = os.environ['IBM_API_KEY']

def get(raw_text):
    """
    Fetches tone information from IBM tone API for given text

    Args:
        raw_text (str): Text to analyze

    Returns:
        Raw API response object on success. Error on failure.
    """
    authenticator = IAMAuthenticator(IBM_API_KEY)
    tone_analyzer = ToneAnalyzerV3(
        version='2020-02-25',
        authenticator=authenticator
    )
    tone_analyzer.set_service_url('https://api.us-south.tone-analyzer.watson.cloud.ibm.com/instances/39cb10a6-c500-45b2-8481-053a17155502')
    try:
        resp = tone_analyzer.tone(
            {'text': raw_text},
            content_type='application/json',
            sentences=False
        )
        if resp.status_code == 200:
            return resp.result
        else:
            return {'error': resp.status_code}
    except Exception as e:
        return {'error': e}

def extract_score_from_tones(tones, tone_id):
    """
    Extracts specific tone score from array of tone values

    Args:
        tones (list): List of tone objects from IBM tone API
        tone_id (str): specific tone id

    Returns:
        Score as float if it exists. NaN if not.
    """
    matching_tones = [t for t in tones if t['tone_id'] == tone_id]

    if not matching_tones:
        return np.nan

    return matching_tones[0]['score']

def extract_score(raw_tone, tone_id):
    """
    Extracts specific tone score from raw tone API response

    Args:
        raw_tone (obj): Raw response object from IBM tone API
        tone_id (str): specific tone id

    Returns:
        Score as float if it exists. NaN if not.
    """
    if not raw_tone.get('document_tone'):
        return np.nan

    tones = raw_tone.get('document_tone').get('tones')
    return extract_score_from_tones(tones, tone_id)
