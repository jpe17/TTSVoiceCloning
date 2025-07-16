from infer_voice import speak_text, preload_voice

# Preload everything upfront
preload_voice("elonmusk")

# Now all generations are instant
speak_text("Hello, my name is Bes")