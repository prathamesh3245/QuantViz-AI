import asyncio
import websockets
import json

# Set of connected clients
connected_clients = set()

async def alpaca_listener():
    URL = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
    API_KEY = "PK0MWZXKKUWPYWB7RBZS"
    SECRET_KEY = "rkdiBVaTyUxGmrsi7fZaz6zM1uZsSwKX8bUMOz3e"

    try:
        async with websockets.connect(URL) as alpaca_ws:
            # Authenticate
            await alpaca_ws.send(json.dumps({
                "action": "auth",
                "key": API_KEY,
                "secret": SECRET_KEY
            }))
            print("Auth:", await alpaca_ws.recv())

            # Subscribe to BTC/USD
            await alpaca_ws.send(json.dumps({
                "action": "subscribe",
                "trades": ["BTC/USD"]
            }))

            print("Subscribed to BTC/USD")

            while True:
                try:
                    msg = await alpaca_ws.recv()
                    msg_json = json.loads(msg)

                    if isinstance(msg_json, list) and msg_json and 'p' in msg_json[0]:
                        price = msg_json[0]['p']
                        print(f"Broadcasting: {price}")
                        await broadcast(str(price))

                except Exception as e:
                    print("[Alpaca Listener Error]", e)
                    await asyncio.sleep(2)
    except Exception as outer:
        print("[Connection to Alpaca Failed]", outer)

async def broadcast(message):
    """Send message to all connected clients"""
    dead_clients = set()
    for client in connected_clients:
        try:
            await client.send(message)
        except Exception as e:
            print("[Broadcast Error]", e)
            dead_clients.add(client)
    connected_clients.difference_update(dead_clients)

async def handler(websocket, path):
    print("Frontend connected")
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    except Exception as e:
        print("[Client Wait Closed Error]", e)
    finally:
        connected_clients.discard(websocket)
        print("Frontend disconnected")


async def main():
    print("Starting WebSocket Server on ws://localhost:8000")

    # Start WebSocket server
    await websockets.serve(handler, "localhost", 8000)

    # Run Alpaca listener in background
    asyncio.create_task(alpaca_listener())

    # Keep the server running forever
    await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user.")
