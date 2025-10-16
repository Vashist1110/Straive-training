import redis

try:
    # Connect to Redis server
    r = redis.Redis(host='localhost', port=6379, db=0)
    # Check connection
    if r.ping():
        print("✅ Connected to Redis!")
    # ---------------------------
    # Set and Get Operations
    # ---------------------------
    r.set('name', 'Alice')
    print("Name stored in Redis:", r.get('name').decode('utf-8'))
    # ---------------------------
    # Working with Numbers
    # ---------------------------
    r.set('count', 10)
    r.incr('count')  # Increment by 1
    print("Updated count:", r.get('count').decode('utf-8'))
    # ---------------------------
    # Storing a List
    # ---------------------------
    r.lpush('tasks', 'task1')
    r.lpush('tasks', 'task2')
    tasks = r.lrange('tasks', 0, -1)
    print("Tasks in Redis list:", [t.decode('utf-8') for t in tasks])
    # ---------------------------
    # Deleting a Key
    # ---------------------------
    r.delete('name')
    print("Deleted key 'name'? Exists:", r.exists('name'))
except redis.ConnectionError as e:
    print(" Redis connection failed:", e)
