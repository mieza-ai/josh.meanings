(ns mieza.meanings.protocols.classifier)

(defprotocol Classifier
  (assignments [this datasets]))