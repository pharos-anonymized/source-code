import json
from dataclasses import asdict
from pathlib import Path

from core.log_data import LogData

data_dir = Path("data")
output_dir = data_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)
json_files = list(data_dir.glob("*.json"))

for json_file in json_files:
    with open(json_file, "r") as f:
        raw_data = json.load(f)
    logdata = LogData.from_json(raw_data)

    humans, agents, buildings = logdata.get_state(min(logdata.timestamps))
    logdata_firstframe = LogData(humans=humans, devices=agents, buildings=buildings)

    output_file = output_dir / json_file.name
    with open(output_file, "w") as f:
        json.dump(asdict(logdata_firstframe), f)
