

its the new rasa

actions has actions
credentials has the connections like slack, fb etc

all is same as before

conda activate rasa_custom

'rasa train nlu'  trains the nlu  part, sees the confog.yml for details

'rasa train' trains everything, againsees the config.yml

'rasa run actions' runs the actions file

'rasa shell' starts for commandline interface

'rasa run' runs the server, basically what run_app.py did.

./ngrok http 5004

activate credentials in slack and chat on slack
