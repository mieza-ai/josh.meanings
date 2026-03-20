(ns mieza.meanings.protocols.savable)

(defprotocol Savable
  (save-model [this filename]))
