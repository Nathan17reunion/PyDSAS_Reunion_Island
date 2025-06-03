# PyDSAS_Reunion_Island

Voici une explication brève pour la création des sous dossier utile pour le script principale

```mermaid
flowchart TD
root --> input_data
input_data --> sds
input_data --> boundary_reunion
root --> aux_data
root --> output_data

Le "sds" est le dossie pour avoir stocké les fichier trait de côte directement les fichier shapefile en cinq format pour un trait de côte (.dbf, .shp, .cpg, .prj, .shx) et optiennel le .qmd 
Boudari_reunion est un dossier pour enregistrer le frontière de la littorale de la réunion
Aux_data un dossier qui contient les unité des cellules hydrosédimentaire de chaque unité morphologique



