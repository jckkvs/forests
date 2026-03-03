import re
import os

files = [
    r'c:\Users\horie\trees\forests\tests\test_builder.py',
    r'c:\Users\horie\trees\forests\examples\02_forest_builder.py',
    r'c:\Users\horie\trees\forests\README.md'
]

for fpath in files:
    if not os.path.exists(fpath):
        continue
        
    with open(fpath, 'r', encoding='utf-8') as f:
        c = f.read()

    # Imports
    c = c.replace('from forests import ForestBuilder, IncompatibleOptionsWarning', 'from forests import ForestsClassifier, ForestsRegressor, IncompatibleOptionsWarning')
    c = c.replace('from forests import ForestBuilder', 'from forests import ForestsClassifier, ForestsRegressor')
    
    # Replace the class name in test_builder.py
    c = c.replace('TestForestBuilderClassification', 'TestForestsClassifier')
    c = c.replace('TestForestBuilderRegression', 'TestForestsRegressor')
    
    # Classification
    c = re.sub(r'ForestBuilder\(([^)]*?),\s*task="classification"(.*?)\)', r'ForestsClassifier(\1\2)', c, flags=re.DOTALL)
    
    # Regression
    c = re.sub(r'ForestBuilder\(([^)]*?),\s*task="regression"(.*?)\)', r'ForestsRegressor(\1\2)', c, flags=re.DOTALL)
    
    # Anything remaining with classification
    c = re.sub(r'ForestBuilder\((.*?task="classification".*?)\)', lambda m: 'ForestsClassifier(' + m.group(1).replace(', task="classification"', '').replace('task="classification", ', '').replace('task="classification"', '') + ')', c, flags=re.DOTALL)
    
    # Anything remaining with regression
    c = re.sub(r'ForestBuilder\((.*?task="regression".*?)\)', lambda m: 'ForestsRegressor(' + m.group(1).replace(', task="regression"', '').replace('task="regression", ', '').replace('task="regression"', '') + ')', c, flags=re.DOTALL)

    # Replace remaining ForestBuilder in README/examples where task wasn't explicitly stated
    c = c.replace('ForestBuilder', 'ForestsClassifier')
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(c)

print('Done!')
