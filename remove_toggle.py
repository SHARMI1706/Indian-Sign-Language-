import re

path = r'c:\project\isl-translator - Copy\templates\index.html'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

before = len(content)

# Remove the entire Speak ON/OFF toggle block
# Matches from the blank line before the comment through the closing </div> and trailing blank line
content = re.sub(
    r'\r?\n\s*<!-- Speak ON/OFF toggle -->.*?</div>(\r?\n\s*\r?\n)',
    r'\1',
    content,
    flags=re.DOTALL
)

after = len(content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

if after < before:
    print(f'Done! Removed {before - after} chars. File went from {before} to {after} bytes.')
else:
    print('WARNING: Nothing was removed. Toggle block may already be gone or pattern did not match.')
    print('Searching for "Speak ON/OFF toggle" in file...')
    if 'Speak ON/OFF toggle' in content:
        print('Found it - regex pattern needs adjustment')
    else:
        print('Not found - already removed!')
