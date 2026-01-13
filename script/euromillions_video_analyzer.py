#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'analyse vidéo pour l'analyseur Euromillions.
Ce module permet de visualiser et d'analyser les vidéos de tirage Euromillions.
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("EuromillionsVideoAnalyzer")

class EuromillionsVideoAnalyzer:
    """Classe pour l'analyse des vidéos de tirage Euromillions."""
    
    def __init__(self, video_dir=None, output_dir=None):
        """
        Initialise le module d'analyse vidéo.
        
        Args:
            video_dir: Répertoire contenant les vidéos de tirage (optionnel)
            output_dir: Répertoire de sortie pour les résultats d'analyse (optionnel)
        """
        self.video_dir = Path(video_dir) if video_dir else Path("tirage_video")
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("resultats_euromillions_video")
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Répertoire créé: {self.output_dir}")
        
        # Attributs pour la vidéo en cours
        self.current_video_path = None
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.duration = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self.play_thread = None
        
        # Attributs pour l'interface graphique
        self.root = None
        self.frame = None
        self.canvas = None
        self.slider = None
        self.play_button = None
        self.status_label = None
        self.frame_label = None
        self.time_label = None
        
        # Attributs pour l'analyse des boules
        self.detected_numbers = []
        self.detected_stars = []
        
        logger.info(f"EuromillionsVideoAnalyzer initialisé avec video_dir: {self.video_dir}")
    
    def list_videos(self) -> List[Path]:
        """
        Liste toutes les vidéos disponibles dans le répertoire vidéo.
        
        Returns:
            List[Path]: Liste des chemins des vidéos
        """
        if not self.video_dir.exists():
            logger.warning(f"Le répertoire vidéo {self.video_dir} n'existe pas.")
            return []
        
        video_extensions = ['.mp4', '.avi', '.webm', '.mkv', '.mov']
        videos = []
        
        for ext in video_extensions:
            videos.extend(list(self.video_dir.glob(f"*{ext}")))
        
        logger.info(f"Trouvé {len(videos)} vidéos dans {self.video_dir}")
        return videos
    
    def load_video(self, video_path: Union[str, Path]) -> bool:
        """
        Charge une vidéo pour analyse.
        
        Args:
            video_path: Chemin de la vidéo à charger
            
        Returns:
            bool: True si la vidéo a été chargée avec succès, False sinon
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"La vidéo {video_path} n'existe pas.")
            return False
        
        try:
            # Fermer la vidéo précédente si elle existe
            if self.cap is not None:
                self.cap.release()
            
            # Ouvrir la nouvelle vidéo
            self.cap = cv2.VideoCapture(str(video_path))
            if not self.cap.isOpened():
                logger.error(f"Impossible d'ouvrir la vidéo {video_path}")
                return False
            
            # Récupérer les informations de la vidéo
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0
            self.current_frame_idx = 0
            self.current_video_path = video_path
            
            logger.info(f"Vidéo chargée: {video_path}")
            logger.info(f"Nombre de frames: {self.frame_count}")
            logger.info(f"FPS: {self.fps}")
            logger.info(f"Durée: {self.duration:.2f} secondes")
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la vidéo: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def get_frame(self, frame_idx: Optional[int] = None) -> Tuple[bool, np.ndarray]:
        """
        Récupère une frame spécifique de la vidéo.
        
        Args:
            frame_idx: Indice de la frame à récupérer (optionnel, utilise la frame courante si None)
            
        Returns:
            Tuple[bool, np.ndarray]: (succès, frame)
        """
        if self.cap is None:
            logger.error("Aucune vidéo n'est chargée.")
            return False, None
        
        if frame_idx is not None:
            if frame_idx < 0 or frame_idx >= self.frame_count:
                logger.error(f"Indice de frame invalide: {frame_idx}")
                return False, None
            
            # Positionner la vidéo à la frame demandée
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.current_frame_idx = frame_idx
        
        # Lire la frame
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Erreur lors de la lecture de la frame.")
            return False, None
        
        # Si on a lu une frame spécifique, on incrémente l'indice courant
        if frame_idx is None:
            self.current_frame_idx += 1
        
        return True, frame
    
    def detect_balls(self, frame: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Détecte les boules de l'Euromillions dans une frame.
        
        Args:
            frame: Frame à analyser
            
        Returns:
            Tuple[List[int], List[int]]: (numéros principaux, étoiles)
        """
        # Cette fonction est un placeholder pour l'implémentation réelle de détection
        # Elle devrait utiliser des techniques de vision par ordinateur pour détecter les boules
        # et reconnaître les numéros
        
        logger.info("Détection des boules (fonction placeholder)...")
        
        # Placeholder: retourne des numéros aléatoires pour démonstration
        import random
        main_numbers = sorted(random.sample(range(1, 51), 5))
        stars = sorted(random.sample(range(1, 13), 2))
        
        logger.info(f"Numéros détectés: {main_numbers}")
        logger.info(f"Étoiles détectées: {stars}")
        
        return main_numbers, stars
    
    def analyze_video(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyse complète d'une vidéo de tirage.
        
        Args:
            video_path: Chemin de la vidéo à analyser
            
        Returns:
            Dict[str, Any]: Résultats de l'analyse
        """
        if not self.load_video(video_path):
            return {"success": False, "error": "Échec du chargement de la vidéo"}
        
        try:
            logger.info(f"Analyse de la vidéo: {video_path}")
            
            # Initialiser les résultats
            results = {
                "success": True,
                "video_path": str(video_path),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "frame_count": self.frame_count,
                "fps": self.fps,
                "duration": self.duration,
                "main_numbers": None,
                "stars": None,
                "frames": {
                    "start": None,
                    "main_numbers": None,
                    "stars": None,
                    "end": None
                }
            }
            
            # Analyse simplifiée: on prend quelques frames clés pour l'analyse
            key_frames = [
                int(self.frame_count * 0.1),  # Début
                int(self.frame_count * 0.5),  # Milieu (numéros principaux)
                int(self.frame_count * 0.7),  # Étoiles
                int(self.frame_count * 0.9)   # Fin
            ]
            
            # Récupérer et analyser les frames clés
            for i, frame_idx in enumerate(key_frames):
                ret, frame = self.get_frame(frame_idx)
                if not ret:
                    continue
                
                # Sauvegarder la frame
                frame_path = self.output_dir / f"{Path(video_path).stem}_frame_{i}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                # Analyser la frame selon sa position
                if i == 0:
                    results["frames"]["start"] = str(frame_path)
                elif i == 1:
                    main_numbers, _ = self.detect_balls(frame)
                    results["main_numbers"] = main_numbers
                    results["frames"]["main_numbers"] = str(frame_path)
                elif i == 2:
                    _, stars = self.detect_balls(frame)
                    results["stars"] = stars
                    results["frames"]["stars"] = str(frame_path)
                elif i == 3:
                    results["frames"]["end"] = str(frame_path)
            
            # Sauvegarder les résultats
            results_path = self.output_dir / f"{Path(video_path).stem}_analysis.txt"
            with open(results_path, 'w', encoding='utf-8') as f:
                f.write(f"=== ANALYSE VIDÉO EUROMILLIONS ===\n")
                f.write(f"Vidéo: {video_path}\n")
                f.write(f"Date d'analyse: {results['timestamp']}\n")
                f.write(f"Nombre de frames: {results['frame_count']}\n")
                f.write(f"FPS: {results['fps']}\n")
                f.write(f"Durée: {results['duration']:.2f} secondes\n\n")
                f.write(f"=== RÉSULTATS ===\n")
                f.write(f"Numéros principaux: {results['main_numbers']}\n")
                f.write(f"Étoiles: {results['stars']}\n\n")
                f.write(f"=== FRAMES CLÉS ===\n")
                for key, path in results["frames"].items():
                    if path:
                        f.write(f"{key}: {path}\n")
            
            logger.info(f"Analyse terminée. Résultats sauvegardés dans {results_path}")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la vidéo: {str(e)}")
            logger.debug(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def start_gui(self):
        """
        Lance l'interface graphique pour la visualisation des vidéos.
        """
        try:
            # Créer la fenêtre principale
            self.root = tk.Tk()
            self.root.title("Analyseur Vidéo EuroMillions")
            self.root.geometry("1200x800")
            
            # Frame principale
            self.frame = ttk.Frame(self.root, padding=10)
            self.frame.pack(fill=tk.BOTH, expand=True)
            
            # Frame pour les contrôles
            control_frame = ttk.Frame(self.frame)
            control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
            
            # Liste des vidéos
            video_frame = ttk.LabelFrame(self.frame, text="Vidéos disponibles")
            video_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
            
            videos = self.list_videos()
            video_listbox = tk.Listbox(video_frame, width=40, height=20)
            for video in videos:
                video_listbox.insert(tk.END, video.name)
            video_listbox.pack(fill=tk.BOTH, expand=True)
            
            # Bouton pour charger une vidéo
            def load_selected_video():
                selection = video_listbox.curselection()
                if selection:
                    video_name = video_listbox.get(selection[0])
                    video_path = self.video_dir / video_name
                    self.load_video(video_path)
                    self.update_gui()
            
            load_button = ttk.Button(video_frame, text="Charger la vidéo", command=load_selected_video)
            load_button.pack(pady=5)
            
            # Bouton pour analyser une vidéo
            def analyze_selected_video():
                selection = video_listbox.curselection()
                if selection:
                    video_name = video_listbox.get(selection[0])
                    video_path = self.video_dir / video_name
                    results = self.analyze_video(video_path)
                    if results["success"]:
                        tk.messagebox.showinfo("Analyse terminée", 
                                              f"Numéros détectés: {results['main_numbers']}\n"
                                              f"Étoiles détectées: {results['stars']}")
                    else:
                        tk.messagebox.showerror("Erreur", f"Échec de l'analyse: {results.get('error', 'Erreur inconnue')}")
            
            analyze_button = ttk.Button(video_frame, text="Analyser la vidéo", command=analyze_selected_video)
            analyze_button.pack(pady=5)
            
            # Canvas pour afficher la vidéo
            canvas_frame = ttk.LabelFrame(self.frame, text="Visualisation")
            canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            self.canvas = tk.Canvas(canvas_frame, bg="black", width=800, height=450)
            self.canvas.pack(fill=tk.BOTH, expand=True)
            
            # Contrôles de lecture
            self.slider = ttk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL)
            self.slider.pack(fill=tk.X, pady=5)
            
            button_frame = ttk.Frame(control_frame)
            button_frame.pack(fill=tk.X)
            
            self.play_button = ttk.Button(button_frame, text="Lecture", command=self.toggle_play)
            self.play_button.pack(side=tk.LEFT, padx=5)
            
            prev_frame_button = ttk.Button(button_frame, text="< Frame", command=lambda: self.navigate_frames(-1))
            prev_frame_button.pack(side=tk.LEFT, padx=5)
            
            next_frame_button = ttk.Button(button_frame, text="Frame >", command=lambda: self.navigate_frames(1))
            next_frame_button.pack(side=tk.LEFT, padx=5)
            
            # Labels d'information
            info_frame = ttk.Frame(control_frame)
            info_frame.pack(fill=tk.X, pady=5)
            
            self.status_label = ttk.Label(info_frame, text="Aucune vidéo chargée")
            self.status_label.pack(side=tk.LEFT, padx=5)
            
            self.frame_label = ttk.Label(info_frame, text="Frame: 0/0")
            self.frame_label.pack(side=tk.LEFT, padx=5)
            
            self.time_label = ttk.Label(info_frame, text="Temps: 00:00/00:00")
            self.time_label.pack(side=tk.LEFT, padx=5)
            
            # Démarrer la boucle principale
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"Erreur lors du lancement de l'interface graphique: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def update_gui(self):
        """
        Met à jour l'interface graphique avec la frame courante.
        """
        if self.cap is None or self.root is None:
            return
        
        # Récupérer la frame courante
        ret, frame = self.get_frame(self.current_frame_idx)
        if ret and frame is not None:
            # Convertir l'image OpenCV (BGR) en image Tkinter (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk  # Garder une référence pour éviter le garbage collection
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
