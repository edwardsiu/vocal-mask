import requests
from discord import Webhook, RequestsWebhookAdapter, File
import json

with open("discord_config.json", "r") as inf:
    discord_config = json.load(inf)

def get_webhook():
    return Webhook.partial(discord_config["webhook_id"], discord_config["token"], adapter=RequestsWebhookAdapter())

def send_message(msg):
    webhook = get_webhook()
    try:
        webhook.send(msg)
    except:
        pass
    
def send_file(path, msg=None):
    webhook = get_webhook()
    #with open(path, "r") as f:
    webhook.send(content=msg, file=File(path))

def send_files(paths, msg=None):
    files = [File(path) for path in paths]
    webhook = get_webhook()
    try:
        webhook.send(content=msg, files=files)
    except:
        pass
