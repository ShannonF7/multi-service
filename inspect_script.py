import sys
import os
import json

# Add src to path
sys.path.append(os.getcwd())

from src.database.session import engine
from sqlalchemy.orm import sessionmaker
from src.database.models import GeneratedScript

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

script_id = "a39d33d3-4a97-486e-a4a5-8070c305a103"

print(f"Querying script ID: {script_id}")
record = db.query(GeneratedScript).filter(GeneratedScript.id == script_id).first()

if record:
    print(f"Found record created at: {record.created_at}")
    print(f"Team ID: {record.team_id}")
    if record.script:
        print("-" * 20 + " BEGIN JSON " + "-" * 20)
        print(json.dumps(record.script, indent=2, ensure_ascii=False))
        print("-" * 20 + " END JSON " + "-" * 20)
    else:
        print("Script column is None/Empty")
else:
    print("Record not found matching that ID.")

db.close()
