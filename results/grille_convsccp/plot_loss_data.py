import matplotlib.pyplot as plt
import pandas as pd

# 1. Lecture du fichier TSV
df = pd.read_csv('loss_data.tsv', sep='\t')
df['K'] = df['K'].fillna('')

# 2. Configuration de la figure
plt.figure(figsize=(11, 6))

# 3. Groupement et tracé des courbes
for (model, K, ic), group in df.groupby(['model', 'K', 'ic']):
    group = group.sort_values('epoch')
    
    if model == 'SmallUNet':
        label = f"SmallUNet (ic={int(ic)})"
        linestyle = '--'
    else:
        label = f"ScCP (K={int(K)}, ic={int(ic)})"
        linestyle = '-'
        
    plt.plot(
        group['epoch'], 
        group['loss'], 
        label=label, 
        linestyle=linestyle, 
        marker='o', 
        markersize=3,
        alpha=0.8
    )

# On bloque le haut de l'axe à 1.0, le bas s'ajuste automatiquement 
# (car la valeur 0 n'est pas définie sur une échelle log)
plt.ylim(top=0.5, bottom=0.2)
plt.xlim(1, 50)

# Formateurs d'axe pour rendre le log plus lisible (affiche des valeurs décimales au lieu de puissances de 10)
from matplotlib.ticker import FormatStrFormatter
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_minor_formatter(FormatStrFormatter('%.2f'))

plt.xlabel('Époque', fontsize=11)
plt.ylabel('Loss (Échelle Log, plafonnée à 1.0)', fontsize=11)
plt.title('Évolution de la Loss par Époque (Échelle Log)', fontsize=12, fontweight='bold')

# Grille adaptée aux deux niveaux de l'échelle log (major et minor)
plt.grid(True, which="both", linestyle=':', alpha=0.6)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

# 5. Sauvegarde
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Le graphique en échelle log a été sauvegardé sous le nom 'loss_plot.png'.")