"""
Test Translation Functionality
This script tests the translation module independently
"""

try:
    from deep_translator import GoogleTranslator
    print("✅ deep-translator installed successfully\n")
    
    # Test translation function
    def translate_number(digit, target_language='hi'):
        """Translate a digit to word in the target language"""
        english_words = ['zero', 'one', 'two', 'three', 'four', 
                        'five', 'six', 'seven', 'eight', 'nine']
        
        english_word = english_words[int(digit)]
        print(f"📝 English word: {english_word}")
        
        if target_language == 'en':
            return english_word
        
        translator = GoogleTranslator(source='en', target=target_language)
        translated = translator.translate(english_word)
        
        return translated
    
    # Test cases
    print("=" * 60)
    print("TRANSLATION TESTS")
    print("=" * 60)
    
    test_cases = [
        (2, 'hi', 'Hindi'),
        (5, 'ta', 'Tamil'),
        (7, 'te', 'Telugu'),
        (1, 'es', 'Spanish'),
        (9, 'fr', 'French'),
        (3, 'zh-CN', 'Chinese'),
        (4, 'ja', 'Japanese'),
        (0, 'ar', 'Arabic')
    ]
    
    print()
    for digit, lang_code, lang_name in test_cases:
        try:
            result = translate_number(digit, lang_code)
            print(f"✅ Digit {digit} → {lang_name}: {result}")
        except Exception as e:
            print(f"❌ Digit {digit} → {lang_name}: Error - {e}")
        print()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\n💡 Tip: Translation requires internet connection")
    print("💡 Tip: If errors occur, check your internet connection")
    
except ImportError:
    print("❌ deep-translator not installed")
    print("\n📦 To install, run:")
    print("   pip install deep-translator")
    print("\nOr install all requirements:")
    print("   pip install -r requirements.txt")