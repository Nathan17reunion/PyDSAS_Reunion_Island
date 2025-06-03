## PyDSAS_Reunion_Island

Cet projet n'est pas uniquement pour avoir analyser la cinématique du plage mais aussi de valider les trait de côte après avoir faire une détection automatique avec l'outil SAET.

En plus il faut avoir les trait de côte, puis valider et enfin passer à la statistique du modèle DSAS. Pourquoi je n'utilise pas l'outil DSAS qui a une version autonome maintenant pas besoin d'installer sous forme de plugins dans ArcGIS? Parce que je n'ai pas du système windows conforme au installation du DSAS et ArcGIS Pro Desktop. Il y a un autre moyen d'utiliser avec wine sous linux mais il y a un fichier DSAS_6.0.170_mac.dmg qui n'est pas adapter pour le système linux donc je recours à une création de script **PyDSAS**.

Pour le monment je n'ai pas d'assez de connaissance pour prouver la fiabilité de ce résultats de statistique que j'ai obtenues mais les résultats d'après issus de mes traits de côte obtenues à conforme à tous les valeurs statistique que les recherche antérieurs à utiliser. Je suis bassées sur des rapports de BRGM, Observation de littorale de la réunion, CEREMA et des thèse sur la dynamique côtière et la géomorphologie littorale de la réunion.

# TDC

Pour le trait de côte j'ai mis dans un fichier mes trait de côte fusionnée pour une série temporelle d'image de 2016 à 2025 appellé fusion_trait_de_cote.shp
Vous pouvez touver ici aussi le script de test de combinaison des indices que j'inpiré par le script SAET mais pour le moment je n'obtient du resultat. C'est pour un travail de créer un script d'automatisation du détection du trait de côte par utilisation de combinaison d'indices et la seuillage par tatônnement.

Mais pour les résultas inclus dans ma rédaction, j'ai utilisé les résultat obtenus par le test directement de l'outil SAET (Shoreline Automatic Extraction Tool) pour chaque unité morphologique de la littorale réunionnaise.

# Rédaction du rapport de mémoire

Il y a un fichier sous forme de pdf appellé référence bibliographique qui contient tous les instruction que j'utilise pendant les rédanction de mon mémoire. C'est un fichier en général contient un référence bibliographique que j'ai déjà utilisé mais il y a des instruction nécessaire pour une collaboration à la valorisation de ma rédaction de mon mémoire. 

# Cinématique du plage

Voici une explication brève pour la création des sous dossier utile pour le script principale de la calcul des statisques des modèles avec DSAS.

- **sds** : dossier pour stocker les fichiers shapefile du trait de côte (.dbf, .shp, .cpg, .prj, .shx, optionnel .qmd)  
- **boundary_reunion** : dossier pour la frontière littorale de La Réunion  
- **aux_data** : dossier contenant les unités des cellules hydrosédimentaires  
- **output_data** : dossier pour les résultats générés par les scripts  

```mermaid
flowchart TD
    root --> input_data
    input_data --> sds
    input_data --> boundary_reunion
    root --> aux_data
    root --> output_data
