import tkinter as tk
from tkinter import filedialog, ttk
from typing import TypedDict

class FilePathKeyShift(TypedDict):
    file_path: str
    shift_number: int

def get_filepath_keyshift() -> FilePathKeyShift:
    """Create a UI to get file path and shift number from user."""
    root = tk.Tk()
    root.title("Keyshifting with PC-NSF-HiFiGAN")

    tk.Label(root, text="Input File:").grid(row=0, column=0, padx=5, pady=5)
    file_path = tk.StringVar()
    file_entry = tk.Entry(root, textvariable=file_path, width=40)
    file_entry.grid(row=0, column=1, padx=5, pady=5)

    def browse_file():
        filename = filedialog.askopenfilename()
        if filename:
            file_path.set(filename)

    tk.Button(root, text="Browse...", command=browse_file).grid(row=0, column=2, padx=5, pady=5)

    tk.Label(root, text="Shift Number:").grid(row=1, column=0, padx=5, pady=5)
    shift_num = tk.IntVar(value=0)
    spinbox = ttk.Spinbox(
        root,
        from_=-12,
        to=12,
        textvariable=shift_num,
        width=5
    )
    spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    result = {}
    def on_submit():
        result.update({
            'file_path': file_path.get(),
            'shift_num': shift_num.get()
        })
        root.quit()

    tk.Button(root, text="Submit", command=on_submit).grid(row=2, column=1, pady=10)

    root.mainloop()
    root.destroy()

    return result
