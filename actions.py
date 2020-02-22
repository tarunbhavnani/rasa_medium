#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:24:07 2019


@author: tarun.bhavnani@dev.smecorner.com

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import requests
import json
from rasa_core_sdk import Action
from rasa_core_sdk.events import SlotSet
from rasa_core_sdk.forms import FormAction
#from rasa_core.events import UserUtteranceReverted
from rasa_core_sdk.events import UserUtteranceReverted
from rasa_core_sdk.events import UserUttered

from rasa_core_sdk.events import ActionReverted
from rasa_core_sdk.events import FollowupAction
from rasa_core.interpreter import RasaNLUInterpreter
import pandas as pd
import xlrd
import re
#from word2number import w2n
from googletrans import Translator
import nltk
import spacy
import datetime
import requests
nlp= spacy.load("en")


#from final_cibil_los_pull import get_final_data
#from final_cibil_los_pull import elastic_api

logger = logging.getLogger(__name__)



class ActionDefaultFallback(Action):
    def name(self):
        #return "action_question_counter"
        return 'action_default_fallback'
    def run(self, dispatcher, tracker, domain):
        counter= tracker.get_slot('counter')
        current=tracker.get_slot('current')
        interview_state= tracker.get_slot("interview_state")
        last_intent= tracker.latest_message['intent'].get('name')
        last_intent1= tracker.latest_message['intent']
        
        last_message= tracker.latest_message['text']
        user_name= tracker.get_slot("user_name")
        
        logger.info("action_default")
        logger.info(current)
        logger.info(counter)
        logger.info(last_intent1)
        logger.info(last_message)
        
        
        #before interview start
        #interview state turns to "started if details are fetched in action fetch details"
        
        if interview_state == "start":
          last_intent= tracker.latest_message['intent'].get('name')
          if last_intent=="greet":         
            dispatcher.utter_message("Hi, its a strt. How are you!!")
            counter="action_interview_start"
            return[FollowupAction(counter)]
          
          
          #elif counter !="action_interview_start":
          elif last_intent in ["chitchat", "greet", "thank","weather","abuse"]:
            ret="utter_{}".format(last_intent)
            try:
              dispatcher.utter_template(ret, tracker)
              dispatcher.utter_message("Lets start again, how are you!!")
            except:
              dispatcher.utter_message("Do i know u?Lets start again, how are you!!")
              
            counter="action_listen"
            return[FollowupAction(counter)]

        
        #interview starter
        if current==counter=="action_interview_start" and interview_state=="start":
          counter="action_fetch_details"
          return[FollowupAction(counter)]
        
        
        #after interview start
        if interview_state=="started":
          
          
          #check for intents that are out of scope for interview!
          #in these we followup with the same question
          
          #blank reply
          
          if len(last_message)==0:
            dispatcher.utter_message("No Blanks!!, again!!")
            return[FollowupAction(current)]
          
          #repeat
          
          if (last_intent=="repeat") or (last_message=="what") or (last_message=="क्या") or (last_message=="रिपीट"):
            dispatcher.utter_message("मै फिर पूछूँगी.")           
            return[FollowupAction(current)]
          
          #chitchat
          
          if last_intent in ["chitchat", "greet", "thank","weather","abuse", "ask_name", "tell_name"]:
              ret="utter_{}".format(last_intent)
              #dispatcher.utter_message("This is an interview, ill ask again!!")
            
              dispatcher.utter_template(ret, tracker)
              
              return[FollowupAction(current)]
          
          
        
          #goodbye
          
          if last_intent=="goodbye":
            #dispatcher.utter_message("Goodbye {}".format(user_name))
            #counter="action_stop_check"
            return[FollowupAction("action_stop_check")]
        

          #staring the interview with 12345

          if current=="action_fetch_details":
            user_name= last_message
            return[FollowupAction(counter),SlotSet('user_name', user_name) ]  
        
          #stop
          
          if (last_message=="stop") or (last_intent=="stop") or (last_message=="स्टॉप"):
            #dispatcher.utter_message("Goodbye {}".format(user_name))
            #counter="action_stop_check"
            return[FollowupAction("action_stop_check")]
  
          
          if current=="action_stop_check":
            if last_intent== "affirm" or (last_message=="हां") or (last_message=="हाँ"):
              counter= "action_stop"
              return[FollowupAction(counter),SlotSet('interview_state', "aborted")]
            else:
              dispatcher.utter_message("हम PD जारी रखेंगे।")
              #will use the same current and ask again.
              #note thatw e have put the current in counter in action_stop_check for a recall.
              return[FollowupAction(counter)]  

          
           
          if counter=="end":
               dispatcher.utter_message("आपके समय के लिए धन्यवाद!")
               
               return[FollowupAction("action_stop"),SlotSet('interview_state', "Finished")]
        
        
        return[FollowupAction(counter)]
        

#1)

class Actioninterviewstart(Action):
    def name(self):
        return 'action_interview_start'
    def run(self, dispatcher, tracker, domain):
      counter='action_interview_start'
      current="action_interview_start"
      user_name = tracker.get_slot('user_name')
      user_cell=tracker.get_slot('user_cell')
        
      logger.info("action_interview_start")
      
      if (user_name=="Dear" and user_cell=="none"):
           #dispatcher.utter_message("PD शुरू करने के लिए कृपया रेफरेंस आईडी इनपुट करें, डेमो के लिए 12345 इनपुट करें।")
           dispatcher.utter_message("To start, kindly input your yobo id!.Pres 12345 if you don't have a yobo id!")
           return[FollowupAction("action_listen"),SlotSet('counter', counter),SlotSet('current', current) ]
      else:
           dispatcher.utter_message("Continue plz.")
           return[FollowupAction("action_default_fallback")]
        


#2)

class ActionFetchDetails(Action):
    def name(self):
        return 'action_fetch_details'
    def run(self, dispatcher, tracker, domain):
        #this_action='action_interview_start'
        user_name = tracker.get_slot('user_name')
        user_cell=tracker.get_slot('user_cell')
        last_message= tracker.latest_message['text']
        logger.info("action_fetch_details")

        try:
         n=0
         for i in last_message.split():
                #if i in df['applicant_1_phone'].fillna(0).astype(int).astype(str).values.tolist():
                if i.isdigit() :
                    user_cell=i#i = 1024639
                 
                    try:
                        json_response=get_final_data(int(i))
                        #extract data from DB using this id and put it in "json_response"
                        json_response=get_final_data(int(i))
                    except:
                        dispatcher.utter_message("No DB connectivity")
                    
                    n+=1
         if (user_name=="Dear" and user_cell=="12345"):
                    dispatcher.utter_message("Please type in your name!")
                    return[FollowupAction('action_listen'),SlotSet('user_cell', user_cell),SlotSet('interview_state', "started"),SlotSet('current', "action_fetch_details"),SlotSet('counter', "action_q1")]
                    
         if n>1:
                    dispatcher.utter_message("1 से अधिक सेल की पहचान !. कृपया रजिस्टर्ड सेल नंबर डालिए")
                    user_cell='none'
                    return[FollowupAction('action_interview_start')]
         elif n==1 and str(json_response["applicant_1_first_name"])!='None':
                    #"second change, extracting name"#applicant_1_first_name
                    #user_name=str(df[df.applicant_1_phone==int(user_cell)].last_name.item())  
                    
                    try:
                        translator = Translator()
                        translations = translator.translate([str(json_response["applicant_1_first_name"])], dest='hi')
                        for translation in translations:
                            user_name=translation.text+" जी"
                    except:
                        user_name=str(json_response["applicant_1_first_name"]+" जी")  
                        
                    dispatcher.utter_message("नमस्ते {},एसएमई कॉर्नर पीडी में आपका स्वागत है। इस बातचीत के दौरान मैं आपसे व्यक्तिगत, सिबिल और बैंकिंग जानकारी के कुछ प्रश्न पूछूंगी। इसमें 15 मिनट से ज्यादा का समय नहीं लगेगा। अब हम पीडी शुरू करेंगे!".format(user_name))
                    #return[SlotSet('user_name', user_name),SlotSet('user_cell', user_cell)]
                    #from rasa_core_sdk.events import FollowupAction
                    return[FollowupAction('action_business_kind'),SlotSet('interview_state', "started"),SlotSet('user_name', user_name), SlotSet('user_cell', user_cell), SlotSet('json_response', json_response)]
         else:
                    dispatcher.utter_message("रेफरेंस आईडी रजिस्टर्ड नहीं है। कृपया रजिस्टर्ड आईडी इनपुट करें या अपने  इंटरव्यू को पुनर्निर्धारित करने के लिए SMEcorner हेल्पडेस्क से संपर्क करें। धन्यवाद!")
                    return[FollowupAction('action_interview_start')]
        except:
          dispatcher.utter_message("रेफरेंस आईडी रजिस्टर्ड नहीं है। कृपया रजिस्टर्ड आईडी इनपुट करें या अपने  इंटरव्यू को पुनर्निर्धारित करने के लिए SMEcorner हेल्पडेस्क से संपर्क करें। धन्यवाद!")
          return[FollowupAction('action_interview_start')]




          
        

class ActionStop(Action):
    
    def name(self):
        return "action_stop"
    def run(self, dispatcher, tracker, domain):
      user_name= tracker.get_slot("user_name").split()[0]
      dispatcher.utter_message("Can't help you no more {}.\nWell t'was nice meeting you buddy. Goodbye!! ".format(user_name))
      counter="action_stop"
      current="action_stop"
      return [SlotSet('counter', counter),FollowupAction("action_restart"),SlotSet('current', current) ]
 

class ActionEnd(Action):
    
    def name(self):
        return "action_end"
    def run(self, dispatcher, tracker, domain):
      user_name= tracker.get_slot("user_name").split()[0]
      dispatcher.utter_message("Thanks for your support {}.\n Goodbye!! ".format(user_name))
      counter="action_end"
      current="action_end"
      return [SlotSet('counter', counter),FollowupAction("action_restart"),SlotSet('current', current) ]
 
class ActionStopCheck(Action):
    
    def name(self):
        return "action_stop_check"
    def run(self, dispatcher, tracker, domain):
      #user_name= tracker.get_slot("user_name")
      dispatcher.utter_template("utter_stop_check", tracker)
      #ActionSave.run('action_save',dispatcher, tracker, domain)
      counter=tracker.get_slot("current")
      current="action_stop_check"
      return [FollowupAction("action_listen"),SlotSet('current', current),SlotSet('counter', counter) ]

class ActionQ1(Action):
    
    def name(self):
        return "action_q1"
    def run(self, dispatcher, tracker, domain):
      current="action_q1"
      user_name= tracker.get_slot("user_name").split()[0]
      #dispatcher.utter_message("{}. Goodbye!! ".format(user_name))
      buttons = [{'title': 'Yes', 'payload': 'yes'}, {'title': 'No', 'payload': 'no'}] 
      dispatcher.utter_button_message("Do you earn more than 15 lacks per annum?", buttons)

      counter="action_q1f"
      return [SlotSet('counter', counter),FollowupAction("action_listen"),SlotSet('current', current) ]

class ActionQ1f(Action):
    
    def name(self):
        return "action_q1f"
    def run(self, dispatcher, tracker, domain):
      current="action_q1f"
      last_intent= tracker.latest_message['intent'].get('name')
      last_message= tracker.latest_message['text']
      if (last_message=="yes") or (last_intent=="affirm"):
        #dispatcher.utter_message("Do you have any tie-ups with Swiggy, UberEats, Zomato etc. Please specify.")
        counter="action_q2"
        salary="15+"
        return [SlotSet('current', current),SlotSet('counter', counter),FollowupAction(counter),SlotSet('salary', salary)]
      elif (last_message=="no") or (last_intent=="deny"):
        counter="action_stop"
        salary="15-"
        return [SlotSet('current', current),SlotSet('counter', counter),FollowupAction(counter),SlotSet('salary', salary)]
      else:
        dispatcher.utter_message("Please be a little more clear, I'll ask again!")
        counter= "action_q1"
        return [SlotSet('current', current),SlotSet('counter', counter),FollowupAction(counter)]

class ActionQ2(Action):
    
    def name(self):
        return "action_q2"
    def run(self, dispatcher, tracker, domain):
      current="action_q2"
      user_name= tracker.get_slot("user_name").split()[0]
      #dispatcher.utter_message("{}. Goodbye!! ".format(user_name))
      buttons = [{'title': 'Yes', 'payload': 'yes'}, {'title': 'No', 'payload': 'no'}] 
      dispatcher.utter_button_message("Have you studies from one of the IIts or the IIMs?", buttons)

      counter="action_q2f"
      return [SlotSet('counter', counter),FollowupAction("action_listen"),SlotSet('current', current) ]

class ActionQ2f(Action):
    
    def name(self):
        return "action_q2f"
    def run(self, dispatcher, tracker, domain):
      current="action_q2f"
      last_intent= tracker.latest_message['intent'].get('name')
      last_message= tracker.latest_message['text']
      if (last_message=="yes") or (last_intent=="affirm"):
        #dispatcher.utter_message("Do you have any tie-ups with Swiggy, UberEats, Zomato etc. Please specify.")
        counter="action_end"
        education="IIT-IIM"
        return [SlotSet('current', current),SlotSet('counter', counter),FollowupAction(counter),SlotSet('education', education)]
      elif (last_message=="no") or (last_intent=="deny"):
        counter="action_stop"
        education='Uneducated'
        return [SlotSet('current', current),SlotSet('counter', counter),FollowupAction(counter),SlotSet('education', education)]
      else:
        dispatcher.utter_message("Please be a little more clear, I'll ask again!")
        counter= "action_q2"
        return [SlotSet('current', current),SlotSet('counter', counter),FollowupAction(counter)]
