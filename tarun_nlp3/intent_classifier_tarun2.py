from rasa.nlu.components import Component
from rasa.nlu import utils
from rasa.nlu.model import Metadata

import nltk
from tarun_nlp.models import tarun_nlp
import os
from mtranslate import translate

#intent_classifier_tarun.TarunAnalyzer
class TarunAnalyzer(Component):
    

    name = "intent_classifier_tarun"

    provides = ["intent", "intent_ranking"]


    def __init__(self, component_config=None):
        super(TarunAnalyzer, self).__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        """Not needed, because the the model is pretrained"""
        pass



    def convert_to_rasa(self, intents, probabilities, ranking, texts):
        """Convert model output into the Rasa NLU compatible output format."""
        intent = {"name": intents[0], "confidence": probabilities[0]}

        #intent_ranking = [{"name": intent_name, "confidence": score} for intent_name, score in ranking]
        intent_ranking = [{"name": i, "confidence": ranking[i]} for i in ranking]
        text={"name":texts}
        #entity = {"value": value,
        #          "confidence": confidence,
        #          "entity": "sentiment",
        #          "extractor": "sentiment_extractor"}

        return intent, intent_ranking, text


    def process(self, message, **kwargs):
        """Retrieve the text message, pass it to the classifier
            and append the prediction results to the message class."""

        from mtranslate import translate
        texts=translate(message.text)

        intents, probabilities, ranking= tarun_nlp(txt=texts)
        intent, intent_ranking, text=self.convert_to_rasa(intents, probabilities, ranking, texts)

        message.set("intent", [intent], add_to_output=True)
        message.set("intent_ranking", [intent_ranking], add_to_output=True)
        message.set("text", [text], add_to_output=True)

    def persist(self,file_name, model_dir):
        """Pass because a pre-trained model is already persisted"""

        pass

