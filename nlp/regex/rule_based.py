import re

text = "The meeting is scheduled on 21 August 2025 at 10:00 AM."
dates = re.findall(r'\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b', text)
print(dates) # ['21 August 2025']