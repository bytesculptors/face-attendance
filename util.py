import tkinter as tk
from tkinter import messagebox
import face_recognition
import numpy as np
import sqlite3
import json
from sklearn.metrics.pairwise import cosine_similarity


def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="white",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=2,
                        width=20,
                        font=('Helvetica bold', 20)
                    )

    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img):
    # it is assumed there will be at most 1 match in the db

    embeddings_unknown = face_recognition.face_encodings(img)
    print(len(embeddings_unknown))
    print("-----------------------------------")
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    else:
        embeddings_unknown = embeddings_unknown[0]

    conn = sqlite3.connect('face_db.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    records = cursor.fetchall()
    threshold = 0.6
    final_name = 'unknown_person'
    lowest_distance = float('inf')
    
    for _, name, embeddings_json in records:
        embeddings_db = np.array(json.loads(embeddings_json))
        distance = np.linalg.norm(embeddings_unknown -  embeddings_db)
        # similarity = cosine_similarity([embeddings_db], [embeddings_unknown])[0][0]
        print(name, " ", " ", distance)
        print("********************")
        if distance < threshold and distance < lowest_distance:
            final_name = name
            lowest_distance = distance
    conn.close()
    return final_name
    