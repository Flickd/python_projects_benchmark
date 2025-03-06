import json
import logging
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
            logger.info("Configuration loaded successfully from %s", file_path)
            return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Error loading config file: %s", e)
        return {}

def save_config(file_path: str, config: Dict[str, Any]) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(config, file, indent=4)
            logger.info("Configuration saved to %s", file_path)
    except IOError as e:
        logger.error("Error saving config file: %s", e)

if __name__ == "__main__":
    sample_config = {"app_name": "MyApp", "version": "1.0.0", "debug": True}
    save_config("config.json", sample_config)
    loaded_config = load_config("config.json")
    print("Loaded Configuration:", loaded_config) 
