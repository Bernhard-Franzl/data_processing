from database import init_database
from schema import Event, Door
from database import db_session
# init database

init_database()

import datetime as dt
time_stamp = dt.datetime.now()
e = Event(door_id=1, 
            create_at=time_stamp, event_type=1, 
            in_support_count=1, out_support_count=1, 
            sensor_one_support_count=1, sensor_two_support_count=1)
db_session.add(e)
db_session.commit()

print(Event.query.all())

db_session.remove()


#@app.teardown_appcontext
#def shutdown_session(exception=None):
#    db_session.remove()

