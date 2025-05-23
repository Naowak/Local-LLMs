from kokoro import KPipeline
import soundfile as sf
import requests
import os
import json
import re
import numpy as np
import queue
import threading
import simpleaudio as sa
import sys
import warnings

# Remove warnings
import warnings
warnings.filterwarnings("ignore")


# Extract text args
print('Thinking...')
user_question = sys.argv[1]

# Set the environment variable
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib'

# Audio queue for sequential playback
audio_queue = queue.Queue()

# Audio player worker function that runs in a separate thread
def audio_player_worker():
    while True:
        # Get the next audio file from the queue
        audio_data, samplerate = audio_queue.get()
        
        if audio_data is None:  # Signal to stop the thread
            break
            
        # Convert float audio to int16 for playback
        audio_int16 = (audio_data * 32767).numpy().astype(np.int16)
        
        # Play audio and wait for it to complete before playing next
        play_obj = sa.play_buffer(audio_int16, 1, 2, samplerate)
        play_obj.wait_done()
        
        # Mark task as done
        audio_queue.task_done()

# Start the audio player thread
audio_thread = threading.Thread(target=audio_player_worker, daemon=True)
audio_thread.start()

# Initialize pipeline for TTS
pipeline = KPipeline(lang_code='f')  # make sure lang_code matches voice

# Make API request to get the LLM with streaming
response = requests.post(
    "http://tortank.local:2525/v1/chat/completions", 
    json={
        "messages": [{"role": "user", "content": user_question}],
        "temperature": 0.9,
        "stream": True
    },
    stream=True
)

# Initialize buffer to accumulate text
buffer = ""
chunk_counter = 0

# Improved sentence detection pattern
sentence_pattern = re.compile(r'([.!?]|[\n]{2,})')

# Process streaming response
for chunk in response.iter_lines():
    if not chunk:
        continue

    # Decode the chunk
    chunk_text = chunk.decode("utf-8")
    
    # Skip data: prefix if present
    if chunk_text.startswith("data: "):
        chunk_text = chunk_text[6:]
        
    if chunk_text.strip() == "[DONE]" or not chunk_text.strip():
        continue
        
    try:
        # Parse the JSON data
        data = json.loads(chunk_text)
        
        # Extract new text content
        new_text = ""
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                new_text = choice["delta"]["content"]
            elif "text" in choice:
                new_text = choice["text"]
            
        if new_text:
            # print(f"Received: {new_text}", end="", flush=True)
            # Add to buffer
            new_text = new_text.replace('*', '')
            buffer += new_text
            
            # Check for complete sentences
            match = sentence_pattern.search(buffer)
            while match:
                # Process the complete sentence
                sentence_end = match.end()
                complete_sentence = buffer[:sentence_end]
                print()
                buffer = buffer[sentence_end:]
                
                # print(f"\nProcessing: {complete_sentence}")
                
                # Generate audio for the complete sentence
                generator = pipeline(
                    complete_sentence, 
                    voice='ff_siwis',
                    speed=1
                )
                
                for gs, ps, audio in generator:
                    # print(gs)
                    # print(ps)
                    # Save the audio file
                    sf.write(f'chunk_{chunk_counter}.wav', audio, 24000)
                    # Add audio to the playback queue
                    audio_queue.put((audio, 24000))
                    chunk_counter += 1
                
                # Look for more sentences in the remaining buffer
                match = sentence_pattern.search(buffer)
            
    except json.JSONDecodeError:
        print(f"Could not parse JSON: {chunk_text}")

# Process any remaining text
if buffer:
    # print(f"\nProcessing final text: {buffer}")
    generator = pipeline(
        buffer, 
        voice='ff_siwis',
        speed=1
    )
    
    for gs, ps, audio in generator:
        # print(gs)
        # print(ps)
        sf.write(f'chunk_{chunk_counter}.wav', audio, 24000)
        audio_queue.put((audio, 24000))
        chunk_counter += 1

# Wait for all audio to finish playing
audio_queue.join()