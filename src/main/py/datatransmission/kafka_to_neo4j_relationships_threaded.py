import datetime
from kafka import KafkaConsumer
from neo4j import GraphDatabase
import json

neo4j_uri = "bolt://localhost:7688"
neo4j_user = "username"
neo4j_password = "password"
kafka_bootstrap_servers = ['172.17.0.1:9092', '172.17.0.1:9093', '172.17.0.1:9094']
kafka_topic = 'dogecoin-transactions'

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password), max_connection_pool_size=50)

consumer = KafkaConsumer(
    kafka_topic,
    bootstrap_servers=kafka_bootstrap_servers,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='dogecoin-group-neo-relations'
)

def write_to_neo4j_batch(batch):
    with driver.session() as session:
        tx_query = """
        UNWIND $batch AS tx
        MATCH (t:RawTransaction {id: tx.id})
        MATCH (sender:Address {address: tx.from})
        MATCH (receiver:Address {address: tx.to})
        MATCH (block:Block {block: tx.block})
        MATCH (date:Date {date: tx.date})
        MATCH (idNode:TxID {id: tx.id})

        MERGE (sender)-[r:SENT]->(t)
          ON CREATE SET r.amount = tx.value, r.date = tx.date
          ON MATCH  SET r.amount = tx.value, r.date = tx.date

        MERGE (t)-[:RECEIVED]->(receiver)
        MERGE (t)-[:ON_DATE]->(date)
        MERGE (t)-[:IN_BLOCK]->(block)
        MERGE (t)-[:HAS_ID]->(idNode)
        """
        result = session.run(tx_query, batch=batch, timeout=600)
        for record in result:
            ilk_start_time = datetime.datetime.now()
            print(f"Batch zamanı: {ilk_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Toplam batch: {record['batches']}, Toplam işlem: {record['total']}")

if __name__ == "__main__":
    ilk_start_time = datetime.datetime.now()
    print(f"Başlangıç zamanı: {ilk_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    batch = []
    batch_size = 1
    max_messages = 454
    count = 0

    for message in consumer:
        data = message.value
        batch.append(data)
        count += 1

        if len(batch) >= batch_size:

            start_time = datetime.datetime.now()
            print(f"Batch yazma baslagic: {len(batch)} işlem yazılıyor.")
            write_to_neo4j_batch(batch)

            end_time = datetime.datetime.now()
            print(f"Batch yazma süresi: {end_time - start_time} Zaman: {end_time} | {len(batch)} işlem yazıldı.")
            batch = []

        if count >= max_messages:
            break

    if batch:
        start_time = datetime.datetime.now()
        print(f"Batch Kalan yazma baslagic: {len(batch)} işlem yazılıyor.")
        write_to_neo4j_batch(batch)

        end_time = datetime.datetime.now()
        print(f"Batch Kalan yazma süresi: {end_time - start_time} Zaman: {end_time} | {len(batch)} işlem yazıldı.")

    consumer.close()
    end_time = datetime.datetime.now()
    print(f"Bitiş zamanı: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Toplam süre: {end_time - ilk_start_time}")
