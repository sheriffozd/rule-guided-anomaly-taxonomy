import concurrent.futures
import datetime
import json
import requests
from kafka import KafkaProducer

# RPC Ayarları
rpc_user = "rpcUser"
rpc_password = "rcpPassword"
rpc_url = "http://127.0.0.1:22555/"
# Kafka Ayarları
kafka_bootstrap_servers = 'localhost:9092'
kafka_topic = 'doge-tx'

# Kafka Producer oluştur
producer = KafkaProducer(
    bootstrap_servers=kafka_bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# RPC isteği fonksiyonu
def rpc_request(method, params=[]):
    payload = {
        "jsonrpc": "1.0",
        "id": "rpc",
        "method": method,
        "params": params
    }
    response = requests.post(rpc_url, auth=(rpc_user, rpc_password), json=payload)
    return response.json()

# Blok işlemlerini işleyen ve Kafka'ya gönderen fonksiyon
def process_block(height):
    try:
        block_hash = rpc_request("getblockhash", [height])["result"]
        block = rpc_request("getblock", [block_hash, 2])["result"]
    except Exception as e:
        print(f"Hata (Blok {height}): {e}")
        return

    prev_tx_cache = {}
    timestamp = datetime.datetime.utcfromtimestamp(block["time"]).strftime('%Y-%m-%d %H:%M:%S')

    for tx in block["tx"]:
        txid = tx["txid"]
        blockid = block["height"]
        input_addresses = []

        for vin in tx.get("vin", []):
            if "txid" in vin and "vout" in vin:
                try:
                    if vin["txid"] in prev_tx_cache:
                        prev_tx = prev_tx_cache[vin["txid"]]
                    else:
                        prev_tx = rpc_request("getrawtransaction", [vin["txid"], True])["result"]
                        prev_tx_cache[vin["txid"]] = prev_tx

                    vout_info = prev_tx["vout"][vin["vout"]]
                    addresses = vout_info["scriptPubKey"].get("addresses")
                    if addresses:
                        input_addresses.append(addresses[0])
                except Exception:
                    continue

        output_addresses = [
            (vout["scriptPubKey"]["addresses"][0], vout["value"])
            for vout in tx.get("vout", [])
            if "scriptPubKey" in vout and "addresses" in vout["scriptPubKey"]
        ]

        for inp in input_addresses:
            for out_address, value in output_addresses:
                if inp and out_address:
                    # Kafka'ya gönderilecek veri
                    tx_data = {
                        "txid": txid,
                        "from": inp,
                        "to": out_address,
                        "value": value,
                        "date": timestamp,
                        "block": blockid
                    }
                    producer.send(kafka_topic, tx_data)

# Blokları paralel işleyen fonksiyon
def fetch_and_send_to_kafka(start_block, end_block, max_workers=16):
    print(f"Başlangıç Bloğu: {start_block}, Bitiş Bloğu: {end_block}")
    print(f"İşlem Başladı: {datetime.datetime.now()}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_block, range(start_block, end_block))

    producer.flush()
    print(f"Tüm veriler Kafka'ya gönderildi: {datetime.datetime.now()}")


# Ana program
if __name__ == "__main__":
    block_count = 3739999
    start_block = block_count - 15999  # Block between 3739999 and 3724000

    fetch_and_send_to_kafka(start_block, block_count, max_workers=20)
