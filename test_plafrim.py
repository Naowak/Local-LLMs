import asyncio
import concurrent.futures
import requests
import time
import random

NB_REQUESTS = [1, 2, 4, 8, 16, 32, 64, 128]

text = """
Bidirectional Encoder Representations from Transformers (BERT) is a language model based on the transformer architecture,
"""

def fetch_url(i):
    begin = time.time()
    response = requests.post("http://localhost:8000/v1/completions", json={"model": "casperhansen/mixtral-instruct-awq","prompt": text,"max_tokens": 100,"temperature": 0})
    nb_tokens = response.json()['usage']['completion_tokens']
    nb_prompt = response.json()['usage']['prompt_tokens']
    answer = {'nb_tokens': nb_tokens, 'nb_prompt': nb_prompt, 'time': time.time() - begin}
    return answer

async def main(nb):

    with concurrent.futures.ThreadPoolExecutor(max_workers=nb_requests) as executor:

        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor, 
                fetch_url,
                i,
            )
            for i in range(nb_requests)
        ]
        for response in await asyncio.gather(*futures):
            print(response)


for nb_requests in NB_REQUESTS:
    print("Nb requests: ", nb_requests)
    begin = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(nb_requests))
    print("Total time: ", time.time() - begin)

