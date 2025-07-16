from infer_voice import speak_text, preload_voice

# Preload everything upfront
preload_voice("elonmusk")

# Now all generations are instant
speak_text("Hello my name is Bes")
speak_text("After the first text this is instant now?")
speak_text("I hate Donald Trump")