#!/usr/bin/env python3
import os
os.chdir('/Users/chinmayshrivastava/Documents/telcospace/backend')

from dotenv import load_dotenv
load_dotenv()

from app.tasks.map_processing_task import process_floor_plan

project_id = "31455bbe-24fe-4149-adab-e5a5a4f1ba77"
file_path = "/Users/chinmayshrivastava/Documents/telcospace/static/uploads/5a60bd82-64ff-4a02-989a-7106aa1b891d_20251217_140256.png"

print(f"Triggering processing for project {project_id}")
print(f"File path: {file_path}")

result = process_floor_plan.delay(project_id, file_path)
print(f"Task ID: {result.id}")
print("Task submitted!")
