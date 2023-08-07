V tomto programu jsem implementoval hru Racetrack, viz https://en.wikipedia.org/wiki/Racetrack_(game). 
Hru lze spustit rozběhnutím programu racetrack_main.py. 
Nejprve se hráč objeví v menu, kde si může napsat jméno předpřipravené či vlastní dráhy (pomocí BACKSPACE lze umazávat znaky), ENTER dráhu spustí. 
Samotné načítání dráhy (hledání trasy AI) poprvé trvá zhruba 15 vteřin. 
V samotné hře pak hráč může šipkami a klávesou ENTER určovat místo, kam chce, aby formule jela v dalším kroku, klávesou BACKSPACE se může vrátit o krok zpět.
Proti hráči soutěží jedna formulka ovládaná AI. 
Hra končí v případě vyjetí z dráhy (či vzácném případě velkou rychlostí opuštění celého obrázku dráhy v následujícím kole) nebo dojetím do cíle. 
Dojetí do cíle se počítá pokud hráč skončí na libovolném cílovém políčku, či cílovým políčkem projede. 
Po dojetí do cíle (či vyjetí z dráhy) se ukáží výsledky a hráč má možnost klávesou SPACE vybrat novou dráhu či klávesou ENTER znovu projet tu samou.

Požadavky pro spuštění hry (ještě znovu zopakované v requirements.txt) jsou:
pip install --user numpy pillow matplotlib pyglet scikit-image

Hráč si může vytvořit vlastní dráhu například v GIMPu, hra umí načíst libovolnou mapu a přizpůsobit jí okno. 
Mapa by ovšem v ideálním případě neměla být větší než 128x128 pixelů, protože pak už by byla velmi malá. 
Dráha má mít bílou barvu (rgb [210-255, 210-255, 210-255]), start zelenou (rgb [0-40, 210-255, 0-40]) a cíl červenou (rgb [210-255,0-40,0-40]). 
Prostor ohraničující dráhu může mít jakokouliv barvu která nekoliduje z předchozími třemi, např. černou [255, 255, 255]. 