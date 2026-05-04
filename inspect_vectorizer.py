import tensorflow as tf

obj = tf.saved_model.load(r"Models/text_vectorizer")
print(list(obj.signatures.keys()))
for name, fn in obj.signatures.items():
    print(name)
    print(fn.structured_input_signature)
    print(fn.structured_outputs)
