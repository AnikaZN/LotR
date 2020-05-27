import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os

st.title('What Lord of the Rings race are you?')

name = st.text_input(
        'What is your name?', 'Unknown'
)

gender = st.selectbox(
        'What is your gender?',
        ('Female', 'Male', 'Neither')
)

birth = st.selectbox(
        'When were you born?',
        ('Unknown', 'Before the creation of Arda', 'Creation of Arda',
        'Years of the Lamps', 'Years of the Trees', 'First Age', 'Second Age',
        'Third Age', 'Fourth Age')
)

death = st.selectbox(
        'When did you die?',
        ('Unknown', 'Immortal', 'Still Alive', 'First Age', 'Second Age',
        'Third Age', 'Fourth Age')
)

realm = st.selectbox(
        'Where are you from?',
        ('Arnor', 'Arthedain', 'Cirith Ungol', 'Doriath', 'Gondor',
        'Grey Mountains', 'Lonely Mountain', 'Numenor', 'Rohan', 'Shire',
        'Other')
)

spouse = st.selectbox(
        'Are you married?',
        ('No', 'Yes, I have a husband', 'Yes, I have a wife', 'Unknown')
)

hair = st.selectbox(
      'What is your hair color?',
      ('None', 'Golden', 'Black', 'Brown', 'Silver', 'White', 'Red', 'Blonde',
      'Gray', 'Leaves', 'Other')
)

height = st.slider(
        'How tall are you? (in feet)',
        3.0, 18.0
)

df = pd.DataFrame({'name': name, 'gender': gender, 'birth': birth, 'death': death, 'realm': realm, 'spouse': spouse, 'hair': hair, 'height': height}, index = [0])

v = ['a', 'e', 'i', 'o', 'u', 'y', 'A', 'E', 'I', 'O', 'U', 'Y']

def vowel_sum(column):
    chars = []
    for char in name:
        if char in v:
            chars.append(char)
    return len(chars)

df['vowel_count'] = vowel_sum(df['name'])
df['name_length'] = [len(x) for x in df['name']]
df['pct_vowels'] = df['vowel_count'] / df['name_length']
df['first_letter'] = df['name'].str[0]
df['starts_vowel'] = df['first_letter'].isin(v)

model_path = os.path.join('model.pkl')
model = load(model_path)

result = model.predict(df)
output = result[0]

f'You are a member of the race of {output}.'
