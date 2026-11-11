"""Live status file read by the viewer (out/status.json)."""
import json, os, time, fcntl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "out", "status.json")


def _load():
    try:
        return json.load(open(PATH))
    except Exception:
        return dict(activity="", tests={}, order=[], notes=[])


def update(activity=None, test=None, note=None, verdict=None):
    """test = (key, title, state, detail)  with state in pass|warn|fail|running|pending."""
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    lock = open(PATH + ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        s = _load()
        if activity is not None:
            s["activity"] = activity
        if test is not None:
            key, title, state, detail = test
            if key not in s["tests"]:
                s["order"].append(key)
            s["tests"][key] = dict(title=title, state=state, detail=detail)
        if note is not None:
            s["notes"].append(note)
        if verdict is not None:
            s["verdict"] = verdict
        s["updated"] = time.strftime("%H:%M:%S")
        tmp = PATH + ".tmp"
        json.dump(s, open(tmp, "w"), indent=1)
        os.replace(tmp, PATH)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
