# PyDSAS_Reunion_Island

Voici une explication brève pour la création des sous dossier utile pour le script principale

- **sds** : dossier pour stocker les fichiers shapefile du trait de côte (.dbf, .shp, .cpg, .prj, .shx, optionnel .qmd)  
- **boundary_reunion** : dossier pour la frontière littorale de La Réunion  
- **aux_data** : dossier contenant les unités des cellules hydrosédimentaires  
- **output_data** : dossier pour les résultats générés par les scripts  

flowchart TD
root --> input_data
input_data --> sds
input_data --> boundary_reunion
root --> aux_data
root --> output_data




