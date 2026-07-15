import os
import re
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import fitz
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

from functions import (
    get_collection_count,
    extract_pages_from_pdf,
    chunk_pdf_pages,
    add_to_collection,
    query_collection,
    similar_terms,
    clear_collection,
    get_all_embeddings,
)


class PDFViewer:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="both", expand=True)
        self.doc = None
        self.page_num = 0
        self.total_pages = 0
        self.photo = None

        nav = ttk.Frame(self.frame)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀", command=self.prev_page, width=3).pack(side="left")
        self.page_label = ttk.Label(nav, text="No PDF")
        self.page_label.pack(side="left", padx=10)
        ttk.Button(nav, text="▶", command=self.next_page, width=3).pack(side="left")

        self.canvas = tk.Canvas(self.frame, bg="gray80", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())

        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

    def open(self, path, page):
        try:
            self.doc = fitz.open(path)
            self.total_pages = len(self.doc)
            self.page_num = max(1, min(page, self.total_pages))
            self.render()
        except Exception as e:
            self.page_label.configure(text=f"Error: {e}")
            self.doc = None

    def render(self):
        if not self.doc:
            return
        page = self.doc[self.page_num - 1]
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        cw = self.canvas.winfo_width() or 600
        ch = self.canvas.winfo_height() or 400
        scale = min(cw / pix.width, ch / pix.height, 1.5)
        new_w = int(pix.width * scale)
        new_h = int(pix.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self.photo)
        self.page_label.configure(text=f"Page {self.page_num}/{self.total_pages}")

    def prev_page(self):
        if self.doc and self.page_num > 1:
            self.page_num -= 1
            self.render()

    def next_page(self):
        if self.doc and self.page_num < self.total_pages:
            self.page_num += 1
            self.render()


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Document Searcher")
        self.root.geometry("1300x750")
        self.root.minsize(1000, 600)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.build_query_tab()
        self.build_ingest_tab()
        self.build_term_tab()
        self.build_viz_tab()

        self.viz_coords = None
        self.viz_documents = None
        self.viz_metadatas = None
        self.viz_embeddings = None
        self.rect_selector = None

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.count_label = ttk.Label(status_frame, text="")
        self.count_label.pack(side="left")
        ttk.Button(status_frame, text="Clear DB", command=self.clear_db).pack(side="right")
        self.update_count()

    def build_query_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Search")

        paned = ttk.PanedWindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        top = ttk.Frame(left)
        top.pack(fill="x", pady=(5, 5), padx=5)
        ttk.Label(top, text="Query:").pack(side="left")
        self.query_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.query_var, width=50).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(top, text="Search", command=self.search).pack(side="left")

        opts = ttk.Frame(left)
        opts.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(opts, text="Results:").pack(side="left")
        self.n_results_var = tk.StringVar(value="3")
        ttk.Spinbox(opts, from_=1, to=10, textvariable=self.n_results_var, width=4).pack(side="left", padx=5)
        ttk.Label(opts, text="  (double-click a result to view PDF)").pack(side="left")

        self.results_text = tk.Text(left, wrap="word", font=("Courier", 10), cursor="hand2")
        self.results_text.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self.results_text.tag_configure("clickable", foreground="#2563eb", underline=1)
        self.results_text.tag_configure("bg_even", background="#f0f4f8")
        self.results_text.tag_configure("bg_odd", background="#ffffff")
        self.results_text.bind("<Button-1>", self.on_result_click)
        self.results_text.bind("<Key>", lambda e: "break")

        scroll = ttk.Scrollbar(self.results_text, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.viewer = PDFViewer(right)

    def build_ingest_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Ingest PDF")

        form = ttk.Frame(frame)
        form.pack(fill="x", pady=20, padx=20)

        ttk.Label(form, text="PDF File:").grid(row=0, column=0, sticky="w", pady=5)
        self.pdf_path_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.pdf_path_var, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(form, text="Browse", command=self.browse_pdf).grid(row=0, column=2, pady=5)

        ttk.Label(form, text="Chunk Size:").grid(row=1, column=0, sticky="w", pady=5)
        self.chunk_var = tk.StringVar(value="500")
        ttk.Entry(form, textvariable=self.chunk_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(form, text="Overlap:").grid(row=2, column=0, sticky="w", pady=5)
        self.overlap_var = tk.StringVar(value="100")
        ttk.Entry(form, textvariable=self.overlap_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Button(frame, text="Ingest PDF", command=self.ingest).pack(pady=10)

        self.ingest_log = tk.Text(frame, wrap="word", state="disabled", height=8, font=("Courier", 10))
        self.ingest_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def log(self, widget, msg):
        widget.configure(state="normal")
        widget.insert("end", msg + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def browse_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path:
            self.pdf_path_var.set(path)

    def ingest(self):
        path = self.pdf_path_var.get()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Select a valid PDF file.")
            return

        self.log(self.ingest_log, f"Ingesting {os.path.basename(path)}...")

        def task():
            try:
                chunk_size = int(self.chunk_var.get())
                overlap = int(self.overlap_var.get())
            except ValueError:
                self.log(self.ingest_log, "Invalid chunk size or overlap.")
                return

            pages = extract_pages_from_pdf(path)
            if not pages:
                self.log(self.ingest_log, "No text extracted.")
                return

            basename = os.path.splitext(os.path.basename(path))[0]
            ids, documents, metadatas = chunk_pdf_pages(
                pages, chunk_size, overlap, basename, source_path=path
            )

            add_to_collection(documents, metadatas, ids)

            self.log(self.ingest_log, f"Added {len(ids)} chunks.")
            self.update_count()

        threading.Thread(target=task, daemon=True).start()

    def search(self):
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Enter a query.")
            return

        try:
            n = int(self.n_results_var.get())
        except ValueError:
            n = 3

        def task():
            try:
                results = query_collection(query, n_results=n)
            except Exception:
                self.root.after(0, lambda: self.show_results("Collection is empty.", []))
                return

            if results is None:
                self.root.after(0, lambda: self.show_results("Collection is empty.", []))
                return

            ids = results["ids"][0]
            docs = results["documents"][0]
            dists = results["distances"][0]
            metas = results["metadatas"][0]

            data = list(zip(ids, docs, dists, metas))
            self.results_data = data

            self.root.after(0, lambda: self.show_results(query, data))

        threading.Thread(target=task, daemon=True).start()

    def show_results(self, query, data):
        self.results_text.delete("1.0", "end")

        if not data:
            self.results_text.insert("1.0", query)
            return

        header = f"Query: {query}\n{'='*60}\n\n"
        self.results_text.insert("1.0", header)

        for i, (did, doc, dist, meta) in enumerate(data):
            tag = f"result_{i}"
            bg = "bg_even" if i % 2 == 0 else "bg_odd"

            conf = 1 - dist
            page = meta.get("page", "?")
            source = meta.get("source", "?")
            snippet = doc[:500] + ("..." if len(doc) > 500 else "")

            clickable_text = f"[{i+1}] Score: {conf:.4f}  Page: {page}  File: {os.path.basename(source)}"
            body = f"\n    {snippet}\n\n"

            self.results_text.insert("end", clickable_text, (tag, "clickable", bg))
            self.results_text.insert("end", body, (bg,))

    def on_result_click(self, event):
        idx = self.results_text.index(f"@{event.x},{event.y}")
        if not hasattr(self, "results_data") or not self.results_data:
            return "break"

        tags = self.results_text.tag_names(idx)
        for tag in tags:
            if tag.startswith("result_"):
                result_idx = int(tag.split("_")[1])
                _, _, _, meta = self.results_data[result_idx]
                source = meta.get("source", "")
                page = meta.get("page", 1)
                if source and os.path.isfile(source):
                    self.viewer.open(source, page)
                return "break"

        return "break"

    def build_term_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Find Alternatives")

        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(5, 5), padx=5)
        ttk.Label(top, text="Technical term:").pack(side="left")
        self.term_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.term_var, width=40).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(top, text="Find Alternatives", command=self.find_alternatives).pack(side="left")

        opts = ttk.Frame(frame)
        opts.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(opts, text="Results:").pack(side="left")
        self.term_n_var = tk.StringVar(value="5")
        ttk.Spinbox(opts, from_=1, to=20, textvariable=self.term_n_var, width=4).pack(side="left", padx=5)
        ttk.Label(opts, text="  (double-click a result to view PDF)").pack(side="left")

        paned = ttk.PanedWindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        self.term_results_text = tk.Text(left, wrap="word", font=("Courier", 10), cursor="hand2")
        self.term_results_text.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self.term_results_text.tag_configure("clickable", foreground="#2563eb", underline=1)
        self.term_results_text.tag_configure("bg_even", background="#f0f4f8")
        self.term_results_text.tag_configure("bg_odd", background="#ffffff")
        self.term_results_text.tag_configure("exact_header", foreground="#16a34a", font=("Courier", 10, "bold"))
        self.term_results_text.tag_configure("alt_header", foreground="#dc2626", font=("Courier", 10, "bold"))
        self.term_results_text.bind("<Button-1>", self.on_term_result_click)
        self.term_results_text.bind("<Key>", lambda e: "break")

        scroll = ttk.Scrollbar(self.term_results_text, command=self.term_results_text.yview)
        self.term_results_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self.term_viewer = PDFViewer(right)

    def find_alternatives(self):
        term = self.term_var.get().strip()
        if not term:
            messagebox.showwarning("Warning", "Enter a technical term.")
            return

        try:
            n = int(self.term_n_var.get())
        except ValueError:
            n = 5

        def task():
            try:
                results = similar_terms(term, n_results=n)
            except Exception:
                self.root.after(0, lambda: self.show_term_results(term, None))
                return

            self.root.after(0, lambda: self.show_term_results(term, results))

        threading.Thread(target=task, daemon=True).start()

    def show_term_results(self, term, results):
        self.term_results_text.delete("1.0", "end")
        self.term_results_data = []

        if results is None:
            self.term_results_text.insert("1.0", "Collection is empty.")
            return

        exact = results["exact_matches"]
        alts = results["alternative_matches"]

        header = f"Searching for: \"{term}\"\n{'='*60}\n\n"
        self.term_results_text.insert("1.0", header)

        idx = 0

        if exact:
            self.term_results_text.insert("end", "EXACT MATCHES:\n", ("exact_header",))
            self.term_results_text.insert("end", "-" * 40 + "\n")
            for entry in exact:
                tag = f"term_result_{idx}"
                bg = "bg_even" if idx % 2 == 0 else "bg_odd"
                conf = 1 - entry["distance"]
                page = entry["metadata"].get("page", "?")
                source = entry["metadata"].get("source", "?")
                snippet = entry["text"][:500] + ("..." if len(entry["text"]) > 500 else "")

                line = f"[{idx+1}] Score: {conf:.4f}  Page: {page}  File: {os.path.basename(source)}\n"
                self.term_results_text.insert("end", line, (tag, "clickable", bg))
                self.term_results_text.insert("end", f"    {snippet}\n\n", (bg,))
                self.term_results_data.append(entry)
                idx += 1

        if alts:
            self.term_results_text.insert("end", "\nDIFFERENT TERMS USED IN DOCUMENT:\n", ("alt_header",))
            self.term_results_text.insert("end", "-" * 40 + "\n")
            for entry in alts:
                tag = f"term_result_{idx}"
                bg = "bg_even" if idx % 2 == 0 else "bg_odd"
                conf = 1 - entry["distance"]
                page = entry["metadata"].get("page", "?")
                source = entry["metadata"].get("source", "?")
                snippet = entry["text"][:500] + ("..." if len(entry["text"]) > 500 else "")

                line = f"[{idx+1}] Score: {conf:.4f}  Page: {page}  File: {os.path.basename(source)}\n"
                self.term_results_text.insert("end", line, (tag, "clickable", bg))
                self.term_results_text.insert("end", f"    {snippet}\n\n", (bg,))
                self.term_results_data.append(entry)
                idx += 1

        if not exact and not alts:
            self.term_results_text.insert("end", "No results found.")

    def on_term_result_click(self, event):
        idx = self.term_results_text.index(f"@{event.x},{event.y}")
        if not hasattr(self, "term_results_data") or not self.term_results_data:
            return "break"

        tags = self.term_results_text.tag_names(idx)
        for tag in tags:
            if tag.startswith("term_result_"):
                result_idx = int(tag.split("_")[2])
                entry = self.term_results_data[result_idx]
                source = entry["metadata"].get("source", "")
                page = entry["metadata"].get("page", 1)
                if source and os.path.isfile(source):
                    self.term_viewer.open(source, page)
                return "break"

        return "break"

    def build_viz_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Visualize")

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=5, pady=5)
        ttk.Button(top, text="Generate Visualization", command=self.generate_visualization).pack(side="left")
        self.viz_status = ttk.Label(top, text="Click to visualize the vector space.")
        self.viz_status.pack(side="left", padx=10)

        paned = ttk.PanedWindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        paned.add(left, weight=3)

        self.viz_fig = Figure(figsize=(8, 6), dpi=100, facecolor="white")
        self.viz_ax = self.viz_fig.add_subplot(111)
        self.viz_fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=left)
        self.viz_canvas.get_tk_widget().pack(fill="both", expand=True)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        ttk.Label(right, text="Selection Info", font=("Courier", 10, "bold")).pack(pady=(5, 0))
        self.viz_selection_info = ttk.Label(right, text="Draw a rectangle on the plot to select clusters.")
        self.viz_selection_info.pack(padx=5, pady=5)

        ttk.Label(right, text="Top Keywords:", font=("Courier", 9, "bold")).pack(padx=5, anchor="w")
        self.viz_keywords_text = tk.Text(right, wrap="word", state="disabled", height=12, font=("Courier", 9))
        self.viz_keywords_text.pack(fill="x", padx=5, pady=(0, 5))

        ttk.Label(right, text="Chunks in selection:", font=("Courier", 9, "bold")).pack(padx=5, anchor="w")
        chunks_frame = ttk.Frame(right)
        chunks_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self.viz_chunks_text = tk.Text(chunks_frame, wrap="word", state="disabled", font=("Courier", 9))
        self.viz_chunks_text.pack(fill="both", expand=True)
        chunk_scroll = ttk.Scrollbar(self.viz_chunks_text, command=self.viz_chunks_text.yview)
        self.viz_chunks_text.configure(yscrollcommand=chunk_scroll.set)
        chunk_scroll.pack(side="right", fill="y")

    def generate_visualization(self):
        def task():
            try:
                self.root.after(0, lambda: self.viz_status.configure(text="Loading embeddings..."))
                data = get_all_embeddings()

                if data is None:
                    self.root.after(0, lambda: self.viz_status.configure(text="No documents in DB."))
                    return

                ids, documents, embeddings, metadatas = data
                embeddings_array = np.array(embeddings)
                n = len(embeddings_array)

                self.root.after(0, lambda: self.viz_status.configure(text=f"Running t-SNE on {n} chunks..."))

                perplexity = min(30, max(2, n - 1))
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000, learning_rate="auto", init="pca")
                coords_2d = tsne.fit_transform(embeddings_array)

                self.viz_coords = coords_2d
                self.viz_documents = documents
                self.viz_metadatas = metadatas
                self.viz_embeddings = embeddings_array

                filenames = sorted(set(m.get("filename", "unknown") for m in metadatas))
                cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(filenames), 1))
                color_map = {f: cmap(i / max(len(filenames) - 1, 1)) for i, f in enumerate(filenames)}
                colors = [color_map[m.get("filename", "unknown")] for m in metadatas]

                self.root.after(0, lambda: self._plot_scatter(coords_2d, colors, filenames, n))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.viz_status.configure(text=f"Error: {e}"))

        threading.Thread(target=task, daemon=True).start()

    def _plot_scatter(self, coords, colors, filenames, n):
        self.viz_ax.clear()

        if self.rect_selector is not None:
            self.rect_selector.disconnect_events()
            self.rect_selector = None

        for i, fname in enumerate(filenames):
            mask = [m.get("filename", "unknown") == fname for m in self.viz_metadatas]
            pts = coords[mask]
            short = fname[:35] + ("..." if len(fname) > 35 else "")
            self.viz_ax.scatter(pts[:, 0], pts[:, 1], c=[colors[j] for j, m in enumerate(self.viz_metadatas) if mask[j]], s=5, alpha=0.6, label=short)

        self.viz_ax.set_title(f"{n} chunks from {len(filenames)} document(s)")
        if len(filenames) <= 20:
            self.viz_ax.legend(fontsize=6, loc="best", framealpha=0.7, markerscale=2)

        self.viz_canvas.draw()

        self.rect_selector = RectangleSelector(
            self.viz_ax, self._on_rectangle_select,
            useblit=True, button=[1], interactive=True,
            props=dict(facecolor="royalblue", edgecolor="royalblue", alpha=0.15, linewidth=1),
        )

        self.viz_status.configure(text=f"Done. {n} chunks plotted. Draw a rectangle to inspect a region.")
        self.viz_canvas.draw_idle()

    def _on_rectangle_select(self, eclick, erelease):
        if self.viz_coords is None:
            return

        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])

        mask = (
            (self.viz_coords[:, 0] >= x_min) & (self.viz_coords[:, 0] <= x_max) &
            (self.viz_coords[:, 1] >= y_min) & (self.viz_coords[:, 1] <= y_max)
        )

        indices = [i for i, m in enumerate(mask) if m]
        selected_docs = [self.viz_documents[i] for i in indices]
        selected_meta = [self.viz_metadatas[i] for i in indices]

        self.root.after(0, lambda: self._update_selection_panel(selected_docs, selected_meta))

    def _update_selection_panel(self, docs, metas):
        self.viz_selection_info.configure(text=f"{len(docs)} chunk(s) selected")

        self.viz_keywords_text.configure(state="normal")
        self.viz_keywords_text.delete("1.0", "end")

        if len(docs) >= 3:
            try:
                cleaned = [re.sub(r"[^a-zA-Z0-9\s]", " ", d.lower()) for d in docs]
                vectorizer = TfidfVectorizer(max_features=200, stop_words="english", max_df=0.8, ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(cleaned)
                feature_names = vectorizer.get_feature_names_out()
                mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
                top_indices = mean_tfidf.argsort()[::-1][:15]
                self.viz_keywords_text.insert("1.0", "Top Keywords (TF-IDF):\n\n")
                for rank, idx in enumerate(top_indices, 1):
                    word = feature_names[idx]
                    score = mean_tfidf[idx]
                    self.viz_keywords_text.insert("end", f"  {rank:2d}. {word:<22s} {score:.4f}\n")
            except Exception as e:
                self.viz_keywords_text.insert("1.0", f"Keyword extraction failed: {e}")
        elif len(docs) > 0:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", " ".join(docs).lower())
            from collections import Counter
            self.viz_keywords_text.insert("1.0", "Top Words:\n\n")
            for word, count in Counter(words).most_common(15):
                self.viz_keywords_text.insert("end", f"  {word:<22s} {count}x\n")
        else:
            self.viz_keywords_text.insert("1.0", "No chunks in selection.")

        self.viz_keywords_text.configure(state="disabled")

        self.viz_chunks_text.configure(state="normal")
        self.viz_chunks_text.delete("1.0", "end")
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            page = meta.get("page", "?")
            source = os.path.basename(meta.get("source", "?"))
            snippet = doc[:200].replace("\n", " ") + ("..." if len(doc) > 200 else "")
            self.viz_chunks_text.insert("end", f"[{i+1}] {source} p.{page}\n    {snippet}\n\n")
        self.viz_chunks_text.configure(state="disabled")

    def clear_db(self):
        if not messagebox.askyesno("Confirm", "Delete the entire collection?"):
            return
        try:
            clear_collection()
            self.results_data = []
            self.show_results("", [])
            self.log(self.ingest_log, "Collection cleared.")
            self.update_count()
        except Exception:
            messagebox.showerror("Error", "No collection to delete.")

    def update_count(self):
        count = get_collection_count()
        self.count_label.configure(text=f"Documents in DB: {count}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
