from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast

#  Language detection
language_pipeline = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
text = " 六书 / 六書"
language_result = language_pipeline(text)
detected_language = language_result[0]['label']

#  Mapping language code to desired format
language_mapping = {
    "ar": "ar_AR",
    "bg": "bg_BG",
    "de": "de_DE",
    "el": "el_GR",
    "en": "en_US",
    "es": "es_ES",
    "fr": "fr_FR",
    "hi": "hi_IN",
    "it": "it_IT",
    "ja": "ja_JP",
    "nl": "nl_NL",
    "pl": "pl_PL",
    "pt": "pt_PT",
    "ru": "ru_RU",
    "sw": "sw_KE",
    "th": "th_TH",
    "tr": "tr_TR",
    "ur": "ur_PK",
    "vi": "vi_VN",
    "zh": "zh_CN"
}

if detected_language in language_mapping:
    src_language = language_mapping[detected_language]
else:
    print ("Sorry, can't detect this language.")


# model loading for translation
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


# translate to English
tokenizer.src_lang = src_language
encoded_ar = tokenizer(text, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
decoded_translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(decoded_translation)

