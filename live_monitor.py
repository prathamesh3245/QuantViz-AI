import asyncio
import websockets
import json
from alpaca_trade_api.stream import Stream
import asyncio

YOUR_API_KEY = "PK0MWZXKKUWPYWB7RBZS"
YOUR_SECRET_KEY = "rkdiBVaTyUxGmrsi7fZaz6zM1uZsSwKX8bUMOz3e"
URL = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"


async def connect():
    try:
        async with websockets.connect(URL) as ws:

        # Authenticate
            await ws.send(json.dumps({
                "action": "auth",
                "key": YOUR_API_KEY,
                "secret": YOUR_SECRET_KEY
            }))
            print("Auth response:", await ws.recv())  
            
            # Subscribe to trades
            await ws.send(json.dumps({
                "action": "subscribe",
                "trades": ["BTC/USD"] 
            }))
            
            while True:
                msg = await ws.recv()
                print("Received:", msg) 
    except websockets.ConnectionClosedError as e:
        print("Connection closed unexpectedly:", e)
        
            
async def handle_trade(trade):
    print("Received:", trade)
    # Add this line to broadcast prices:
    await websockets.send(str(trade['p']))  # Send price to HTML

asyncio.run(connect())

