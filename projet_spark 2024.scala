// Databricks notebook source
//Détails du projet : Diagnostic du cancer du sein dans le Wisconsin
// Importer les bibliothèques nécessaires

// COMMAND ----------

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{SQLContext, Row, DataFrame, Column}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.ml.feature.{Imputer, ImputerModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, IndexToString}
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.feature.{Bucketizer, Normalizer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification._
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors
import scala.collection.mutable
import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassificationModel
import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier

// Initialiser une session Spark
 val spark = SparkSession
      .builder
      .appName("Projet_Spark Diagnostic du cancer du sein")
      .config("spark.master", "local")
      .config("spark.eventLog.enabled", false)
      .getOrCreate()
    println("Created Spark Session")

// Importer les implicites Spark pour la manipulation des DataFrames

  val sparkImplicits = spark
import sparkImplicits.implicits._

//Lecture du fichier CSV en tant que DataFrame
val data_f = (sqlContext.read
                  .option("header","true")
                  .option("inferSchema","true")
                  .format("csv")
                  .load("/FileStore/tables/data.csv")
)


// COMMAND ----------

// Afficher le schéma du DataFrame pour voir la structure des données

data_f.printSchema()


// COMMAND ----------

//Afficher le DataFrame sous forme de table
display(data_f)


// COMMAND ----------

//Définir une fonction UDF pour vérifier si 90% ou plus des valeurs d'une ligne sont nulles

val isMostlyNull = udf { xs: Seq[String] =>
  xs.count(_ == null) >= (xs.length * 0.9)
}

//Convertir les colonnes du DataFrame en un tableau de colonnes 
val columns = array(data_f.columns.map(col): _*)

//Filtrer les lignes où moins de 90% des valeurs sont nulles avec l'affichage du  statistiques descriptives
val filteredSummary = data_f.filter(not(isMostlyNull(columns))).describe()

//Affichage de statistiques descriptives du DataFrame filtré
display(filteredSummary)


// COMMAND ----------

// Ajouter une nouvelle colonne "Int_Diagnostics" au DataFrame "data_f"
// La colonne "Int_Diagnostics" sera 1 si "diagnosis" est "M" (Malignant) et 0 sinon (Benign)
val DF_features =  data_f.withColumn("Int_Diagnostics", when($"diagnosis" === "M",1).otherwise(0))

// Affichage le DataFrame + la nouvelle colonne ajoutée
display(DF_features)


// COMMAND ----------

// Creation vue temporaire nommée "CancerSein_Data" à partir du DataFrame `DF_features`

DF_features.createOrReplaceTempView("CancerSein_Data")


// COMMAND ----------

// Sélection des caractéristiques mean data

val mean_features_data = spark.sql("Select diagnosis, radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,`concave points_mean`,symmetry_mean,fractal_dimension_mean from CancerSein_Data")


// COMMAND ----------

//Affichage du mean_features_data
display(mean_features_data)


// COMMAND ----------

// Sélection des caractéristiques se data

val se_features_data = spark.sql("Select diagnosis,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,`concave points_se`,symmetry_se,fractal_dimension_se from CancerSein_Data")


// COMMAND ----------

//Affichage du se_features_data

display(se_features_data)


// COMMAND ----------

// Sélection des caractéristiques worst data

val worst_features_data = spark.sql("Select diagnosis, radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,`concave points_worst`,symmetry_worst,fractal_dimension_worst from CancerSein_Data")


// COMMAND ----------

//Affichage du worst_features_data

display(worst_features_data)


// COMMAND ----------

// Filtrer les données pour ne conserver que les tumeurs bénignes

val DF_Benign = DF_features.filter(row => row.getAs[String]("diagnosis").contains("B") )


// COMMAND ----------

// Affichage table des tumeurs bénignes

display(DF_Benign)


// COMMAND ----------

// Créer une vue temporaire pour les données Non Cancer Data => Int_Diagnostics = 0

DF_Benign.createOrReplaceTempView("Non_Cancer_Data")

// COMMAND ----------

// Identifiant du patient 
val identifier = 85713702 

// Extraction des caractéristiques moyennes basées sur l'identifiant du patient (mean)

val radius_mean = spark.sql("Select radius_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val texture_mean = spark.sql("Select texture_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val perimeter_mean = spark.sql("Select perimeter_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val area_mean = spark.sql("Select area_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val smoothness_mean = spark.sql("Select smoothness_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val compactness_mean = spark.sql("Select compactness_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concavity_mean = spark.sql("Select concavity_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concave_points_mean = spark.sql("Select `concave points_mean` from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val symmetry_mean = spark.sql("Select symmetry_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val fractal_dimension_mean = spark.sql("Select fractal_dimension_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble

// Extraction des caractéristiques les plus mauvaises basées sur l'identifiant du patient (worst)
val radius_worst = spark.sql("Select radius_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val texture_worst = spark.sql("Select texture_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val perimeter_worst = spark.sql("Select perimeter_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val area_worst = spark.sql("Select area_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val smoothness_worst = spark.sql("Select smoothness_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val compactness_worst = spark.sql("Select compactness_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concavity_worst = spark.sql("Select concavity_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concave_points_worst = spark.sql("Select `concave points_worst` from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val symmetry_worst = spark.sql("Select symmetry_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val fractal_dimension_worst = spark.sql("Select fractal_dimension_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble


// COMMAND ----------

// Calculez les valeurs moyennes pour  les données mean
val avg_radius = spark.sql("Select avg(radius_mean) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val avg_texture = spark.sql("Select avg(texture_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_perimeter = spark.sql("Select avg(perimeter_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_area = spark.sql("Select avg(area_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_smoothness = spark.sql("Select avg(smoothness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_compactness = spark.sql("Select avg(compactness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_concavity = spark.sql("Select avg(concavity_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_concave_points = spark.sql("Select avg(`concave points_mean`) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_symmetry = spark.sql("Select avg(symmetry_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_fractal_dimension = spark.sql("Select avg(fractal_dimension_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

// Calculez les valeurs moyennes pour  les données worst
val avg_radius_worst = spark.sql("Select avg(radius_worst) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val avg_texture_worst = spark.sql("Select avg(texture_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_perimeter_worst = spark.sql("Select avg(perimeter_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_area_worst = spark.sql("Select avg(area_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_smoothness_worst = spark.sql("Select avg(smoothness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_compactness_worst = spark.sql("Select avg(compactness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_concavity_worst = spark.sql("Select avg(concavity_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_concave_points_worst = spark.sql("Select avg(`concave points_worst`) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_symmetry_worst = spark.sql("Select avg(symmetry_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_fractal_dimension_worst = spark.sql("Select avg(fractal_dimension_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble


// COMMAND ----------

// Calculez les valeurs min  pour les données mean
val min_radius = spark.sql("Select min(radius_mean) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val min_texture = spark.sql("Select min(texture_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_perimeter = spark.sql("Select min(perimeter_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_area = spark.sql("Select min(area_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_smoothness = spark.sql("Select min(smoothness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_compactness = spark.sql("Select min(compactness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concavity = spark.sql("Select min(concavity_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concave_points = spark.sql("Select min(`concave points_mean`) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_symmetry = spark.sql("Select min(symmetry_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_fractal_dimension = spark.sql("Select min(fractal_dimension_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

// Calculez les valeurs min  pour les données worst
val min_radius_worst = spark.sql("Select min(radius_worst) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val min_texture_worst = spark.sql("Select min(texture_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_perimeter_worst = spark.sql("Select min(perimeter_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_area_worst = spark.sql("Select min(area_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_smoothness_worst = spark.sql("Select min(smoothness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_compactness_worst = spark.sql("Select min(compactness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concavity_worst = spark.sql("Select min(concavity_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concave_points_worst = spark.sql("Select min(`concave points_worst`) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_symmetry_worst = spark.sql("Select min(symmetry_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_fractal_dimension_worst = spark.sql("Select min(fractal_dimension_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble


// COMMAND ----------

// Calculez les valeurs max  pour les données mean
val max_radius = spark.sql("Select max(radius_mean) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val max_texture = spark.sql("Select max(texture_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_perimeter = spark.sql("Select max(perimeter_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_area = spark.sql("Select max(area_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_smoothness = spark.sql("Select max(smoothness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_compactness = spark.sql("Select max(compactness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concavity = spark.sql("Select max(concavity_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concave_points = spark.sql("Select max(`concave points_mean`) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_symmetry = spark.sql("Select max(symmetry_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_fractal_dimension = spark.sql("Select max(fractal_dimension_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

// Calculez les valeurs max  pour les données worst
val max_radius_worst = spark.sql("Select max(radius_worst) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val max_texture_worst = spark.sql("Select max(texture_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_perimeter_worst = spark.sql("Select max(perimeter_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_area_worst = spark.sql("Select max(area_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_smoothness_worst = spark.sql("Select max(smoothness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_compactness_worst = spark.sql("Select max(compactness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concavity_worst = spark.sql("Select max(concavity_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concave_points_worst = spark.sql("Select max(`concave points_worst`) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_symmetry_worst = spark.sql("Select max(symmetry_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_fractal_dimension_worst = spark.sql("Select max(fractal_dimension_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble


// COMMAND ----------

// fonction pour la calcule la mise à l'échelle d'une valeur donnée entre un minimum et un maximum spécifiés.

def Calculation(x:Double, min:Double, max:Double ) : Double = {
      var Calc:Double = 0
      Calc = (x - min)/(max - min)

      return Calc
   }


// COMMAND ----------

// Calcul des valeurs mean 
val actual_Radius = Calculation (radius_mean,min_radius,max_radius)
val actual_texture = Calculation (texture_mean,min_texture,max_texture)
val actual_perimeter = Calculation (perimeter_mean,min_perimeter,max_perimeter)
val actual_area = Calculation (area_mean,min_area,max_area)
val actual_smoothness = Calculation (smoothness_mean,min_smoothness,max_smoothness)
val actual_compactness= Calculation (compactness_mean,min_compactness,max_compactness)
val actual_concavity = Calculation (concavity_mean,min_concavity,max_concavity)
val actual_concave_points = Calculation (concave_points_mean,min_concave_points,max_concave_points)
val actual_symmetry = Calculation (symmetry_mean,min_symmetry,max_symmetry)
val actual_fractal_dimension = Calculation (fractal_dimension_mean,min_fractal_dimension,max_fractal_dimension)

// Calcul des valeurs worst 
val actual_Radius_worst = Calculation (radius_worst,min_radius_worst,max_radius_worst)
val actual_texture_worst = Calculation (texture_worst,min_texture_worst,max_texture_worst)
val actual_perimeter_worst = Calculation (perimeter_worst,min_perimeter_worst,max_perimeter_worst)
val actual_area_worst = Calculation (area_worst,min_area_worst,max_area_worst)
val actual_smoothness_worst = Calculation (smoothness_worst,min_smoothness_worst,max_smoothness_worst)
val actual_compactness_worst= Calculation (compactness_worst,min_compactness_worst,max_compactness_worst)
val actual_concavity_worst = Calculation (concavity_worst,min_concavity_worst,max_concavity_worst)
val actual_concave_points_worst = Calculation (concave_points_worst,min_concave_points_worst,max_concave_points_worst)
val actual_symmetry_worst = Calculation (symmetry_worst,min_symmetry_worst,max_symmetry_worst)
val actual_fractal_dimension_worst = Calculation (fractal_dimension_worst,min_fractal_dimension_worst,max_fractal_dimension_worst)


// COMMAND ----------

//Calcul des valeurs mean optimales pour comparer le cancer Malignant et  Benign
val optimum_Radius = Calculation (avg_radius,min_radius,max_radius)
val optimum_texture = Calculation (avg_texture,min_texture,max_texture)
val optimum_perimeter = Calculation (avg_perimeter,min_perimeter,max_perimeter)
val optimum_area = Calculation (avg_area,min_area,max_area)
val optimum_smoothness = Calculation (avg_smoothness,min_smoothness,max_smoothness)
val optimum_compactness= Calculation (avg_compactness,min_compactness,max_compactness)
val optimum_concavity = Calculation (avg_concavity,min_concavity,max_concavity)
val optimum_concave_points = Calculation (avg_concave_points,min_concave_points,max_concave_points)
val optimum_symmetry = Calculation (avg_symmetry,min_symmetry,max_symmetry)
val optimum_fractal_dimension = Calculation (avg_fractal_dimension,min_fractal_dimension,max_fractal_dimension)

//Calcul des valeurs worst optimales pour comparer le cancer Malignant et  Benign
val optimum_Radius_worst = Calculation (avg_radius_worst,min_radius_worst,max_radius_worst)
val optimum_texture_worst = Calculation (avg_texture_worst,min_texture_worst,max_texture_worst)
val optimum_perimeter_worst = Calculation (avg_perimeter_worst,min_perimeter_worst,max_perimeter_worst)
val optimum_area_worst = Calculation (avg_area_worst,min_area_worst,max_area_worst)
val optimum_smoothness_worst = Calculation (avg_smoothness_worst,min_smoothness_worst,max_smoothness_worst)
val optimum_compactness_worst = Calculation (avg_compactness_worst,min_compactness_worst,max_compactness_worst)
val optimum_concavity_worst = Calculation (avg_concavity_worst,min_concavity_worst,max_concavity_worst)
val optimum_concave_points_worst = Calculation (avg_concave_points_worst,min_concave_points_worst,max_concave_points_worst)
val optimum_symmetry_worst = Calculation (avg_symmetry_worst,min_symmetry_worst,max_symmetry_worst)
val optimum_fractal_dimension_worst = Calculation (avg_fractal_dimension_worst,min_fractal_dimension_worst,max_fractal_dimension_worst)


// COMMAND ----------

// Filtrer les données pour ne garder que les lignes où le diagnostic est "M" (Malin)

val DF_Malin = DF_features.filter(row => row.getAs[String]("diagnosis").contains("M") )


// COMMAND ----------

//Acffichage data Malin
display(DF_Malin)


// COMMAND ----------

// Crée une vue temporaire nommée "Cancer_Data" à partir DF_Malin pour permettre des requêtes SQL personnalisées

DF_Malin.createOrReplaceTempView("Cancer_Data")


// COMMAND ----------

val identifier_2 = 84358402 

// Sélectionner les  valeurs mean selon  l'identifiant 
val radius_mean_m = spark.sql("Select radius_mean from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val texture_mean_m = spark.sql("Select texture_mean from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val perimeter_mean_m = spark.sql("Select perimeter_mean from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val area_mean_m = spark.sql("Select area_mean from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val smoothness_mean_m = spark.sql("Select smoothness_mean from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val compactness_mean_m = spark.sql("Select compactness_mean from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concavity_mean_m = spark.sql("Select concavity_mean from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concave_points_mean_m = spark.sql("Select `concave points_mean` from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val symmetry_mean_m = spark.sql("Select symmetry_mean from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val fractal_dimension_mean_m = spark.sql("Select fractal_dimension_mean from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble

// Sélectionner les  valeurs worst selon   l'identifiant 
val radius_worst_m = spark.sql("Select radius_worst from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val texture_worst_m = spark.sql("Select texture_worst from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val perimeter_worst_m = spark.sql("Select perimeter_worst from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val area_worst_m = spark.sql("Select area_worst from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val smoothness_worst_m = spark.sql("Select smoothness_worst from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val compactness_worst_m = spark.sql("Select compactness_worst from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concavity_worst_m = spark.sql("Select concavity_worst from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concave_points_worst_m = spark.sql("Select `concave points_worst` from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val symmetry_worst_m = spark.sql("Select symmetry_worst from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble
val fractal_dimension_worst_m = spark.sql("Select fractal_dimension_worst from Cancer_Data where id=" + identifier_2).collect.map(_.getDouble(0)).mkString(" ").toDouble


// COMMAND ----------

// Collecte les valeurs min pour mean attributs 
val min_radius_m = spark.sql("Select min(radius_mean) from Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val min_texture_m = spark.sql("Select min(texture_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_perimeter_m = spark.sql("Select min(perimeter_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_area_m = spark.sql("Select min(area_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_smoothness_m = spark.sql("Select min(smoothness_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_compactness_m = spark.sql("Select min(compactness_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concavity_m = spark.sql("Select min(concavity_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concave_points_m = spark.sql("Select min(`concave points_mean`) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_symmetry_m = spark.sql("Select min(symmetry_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_fractal_dimension_m = spark.sql("Select min(fractal_dimension_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

// Collecte les valeurs min pour worst attributs 
val min_radius_worst_m = spark.sql("Select min(radius_worst) from Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val min_texture_worst_m = spark.sql("Select min(texture_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_perimeter_worst_m = spark.sql("Select min(perimeter_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_area_worst_m = spark.sql("Select min(area_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_smoothness_worst_m = spark.sql("Select min(smoothness_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_compactness_worst_m = spark.sql("Select min(compactness_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concavity_worst_m = spark.sql("Select min(concavity_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concave_points_worst_m = spark.sql("Select min(`concave points_worst`) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_symmetry_worst_m = spark.sql("Select min(symmetry_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_fractal_dimension_worst_m = spark.sql("Select min(fractal_dimension_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble


// COMMAND ----------

// Collecte les valeurs max  pour mean attributs 
val max_radius_m = spark.sql("Select max(radius_mean) from Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val max_texture_m = spark.sql("Select max(texture_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_perimeter_m = spark.sql("Select max(perimeter_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_area_m = spark.sql("Select max(area_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_smoothness_m = spark.sql("Select max(smoothness_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_compactness_m = spark.sql("Select max(compactness_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concavity_m = spark.sql("Select max(concavity_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concave_points_m = spark.sql("Select max(`concave points_mean`) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_symmetry_m = spark.sql("Select max(symmetry_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_fractal_dimension_m = spark.sql("Select max(fractal_dimension_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

// Collecte les valeurs max  pour worst attributs 
val max_radius_worst_m = spark.sql("Select max(radius_worst) from Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val max_texture_worst_m = spark.sql("Select max(texture_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_perimeter_worst_m = spark.sql("Select max(perimeter_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_area_worst_m = spark.sql("Select max(area_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_smoothness_worst_m = spark.sql("Select max(smoothness_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_compactness_worst_m = spark.sql("Select max(compactness_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concavity_worst_m = spark.sql("Select max(concavity_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concave_points_worst_m = spark.sql("Select max(`concave points_worst`) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_symmetry_worst_m = spark.sql("Select max(symmetry_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_fractal_dimension_worst_m = spark.sql("Select max(fractal_dimension_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble


// COMMAND ----------

// Calcul des valeurs réelles pour chaque attribut (mean) en utilisant les moyennes, minimums et maximums  
val actual_Radius_m = Calculation (radius_mean_m,min_radius_m,max_radius_m)
val actual_texture_m = Calculation (texture_mean_m,min_texture_m,max_texture_m)
val actual_perimeter_m = Calculation (perimeter_mean_m,min_perimeter_m,max_perimeter_m)
val actual_area_m = Calculation (area_mean_m,min_area_m,max_area_m)
val actual_smoothness_m = Calculation (smoothness_mean_m,min_smoothness_m,max_smoothness_m)
val actual_compactness_m = Calculation (compactness_mean_m,min_compactness_m,max_compactness_m)
val actual_concavity_m = Calculation (concavity_mean_m,min_concavity_m,max_concavity_m)
val actual_concave_points_m = Calculation (concave_points_mean_m,min_concave_points_m,max_concave_points_m)
val actual_symmetry_m = Calculation (symmetry_mean_m,min_symmetry_m,max_symmetry_m)
val actual_fractal_dimension_m = Calculation (fractal_dimension_mean_m,min_fractal_dimension_m,max_fractal_dimension_m)

// Calcul des valeurs réelles pour chaque attribut (worst) en utilisant les moyennes, minimums et maximums
val actual_Radius_worst_m = Calculation (radius_worst_m,min_radius_worst_m,max_radius_worst_m)
val actual_texture_worst_m = Calculation (texture_worst_m,min_texture_worst_m,max_texture_worst_m)
val actual_perimeter_worst_m = Calculation (perimeter_worst_m,min_perimeter_worst_m,max_perimeter_worst_m)
val actual_area_worst_m = Calculation (area_worst_m,min_area_worst_m,max_area_worst_m)
val actual_smoothness_worst_m = Calculation (smoothness_worst_m,min_smoothness_worst_m,max_smoothness_worst_m)
val actual_compactness_worst_m = Calculation (compactness_worst_m,min_compactness_worst_m,max_compactness_worst_m)
val actual_concavity_worst_m = Calculation (concavity_worst_m,min_concavity_worst_m,max_concavity_worst_m)
val actual_concave_points_worst_m = Calculation (concave_points_worst_m,min_concave_points_worst_m,max_concave_points_worst_m)
val actual_symmetry_worst_m = Calculation (symmetry_worst_m,min_symmetry_worst_m,max_symmetry_worst_m)
val actual_fractal_dimension_worst_m = Calculation (fractal_dimension_worst_m,min_fractal_dimension_worst_m,max_fractal_dimension_worst_m)


// COMMAND ----------

// Création de séquences (Radar Graph) pour les valeurs optimum et worst attributes


val rs_w1 = Seq(optimum_fractal_dimension_worst, optimum_texture_worst, optimum_perimeter_worst, optimum_area_worst, optimum_smoothness_worst,   optimum_concavity_worst, optimum_concave_points_worst, optimum_symmetry_worst,optimum_compactness_worst,optimum_Radius_worst )

val rs_w2 = Seq(actual_fractal_dimension_worst, actual_texture_worst, actual_perimeter_worst, actual_area_worst, actual_smoothness_worst,actual_concavity_worst, actual_concave_points_worst, actual_symmetry_worst,actual_compactness_worst,actual_Radius_worst )

val rs_w3 = Seq(actual_fractal_dimension_worst_m, actual_texture_worst_m, actual_perimeter_worst_m, actual_area_worst_m, actual_smoothness_worst_m,  actual_concavity_worst_m, actual_concave_points_worst_m, actual_symmetry_worst_m, actual_compactness_worst_m, actual_Radius_worst_m )

displayHTML(s""" <head>
               <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
               <div style="width: 1100px;">
               <div id="myDiv_1" style="float:left;width:500px;"></div>
               <div id="myDiv_blank" style="float:left;width:100px;"></div>
               <div id="myDiv_2" style="float:left;width:500px;"></div>
               </div>
               <script>
                      data_1 = [
                       {
                       type: 'scatterpolar',
                       r: [${rs_w1(0)}, ${rs_w1(1)}, ${rs_w1(2)}, ${rs_w1(3)}, ${rs_w1(4)}, ${rs_w1(5)}, ${rs_w1(6)}, ${rs_w1(7)}, ${rs_w1(8)}, ${rs_w1(9)}],
                       theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness',  'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Average size of Tumor'
                          
                          },
                          {
                          type: 'scatterpolar',
                          r: [${rs_w2(0)}, ${rs_w2(1)}, ${rs_w2(2)}, ${rs_w2(3)}, ${rs_w2(4)}, ${rs_w2(5)}, ${rs_w2(6)}, ${rs_w2(7)}, ${rs_w2(8)}, ${rs_w2(9)}],
                          theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness', 'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Benign Tumor Patient'
                          }
                        ]
                      
                      data_2 = [
                       {
                       type: 'scatterpolar',
                       r: [${rs_w1(0)}, ${rs_w1(1)}, ${rs_w1(2)}, ${rs_w1(3)}, ${rs_w1(4)}, ${rs_w1(5)}, ${rs_w1(6)}, ${rs_w1(7)}, ${rs_w1(8)}, ${rs_w1(9)}],
                       theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness',  'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Average size of Tumor'
                          
                          },
                          {
                          type: 'scatterpolar',
                          r: [${rs_w3(0)}, ${rs_w3(1)}, ${rs_w3(2)}, ${rs_w3(3)}, ${rs_w3(4)}, ${rs_w3(5)}, ${rs_w3(6)}, ${rs_w3(7)}, ${rs_w3(8)}, ${rs_w3(9)}],
                          theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness', 'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Malignant Cancer Patient'
                          }
                        ]
                        
                      layout_1 = {
                        title : "Analyzing the worst across different dimensions for Benign Tumor",
                        "titlefont": {
                                              family : 'Arial', size:14, color:'#000'
                                     }, 
                          polar: {
                        radialaxis: {
      
                              visible: false,
                              range: [0, 1]
                            }
                          }
                        }
                        
                        layout_2 = {
                        title : "Analyzing the worst across different dimensions for Malignant Cancer",
                        "titlefont": {
                                              family : 'Arial', size:14, color:'#000'
                                     }, 
                          polar: {
                        radialaxis: {
      
                              visible: false,
                              range: [0, 1]
                            }
                          }
                        }

                        Plotly.plot("myDiv_1", data_1, layout_1)
                        Plotly.plot("myDiv_2", data_2, layout_2)
               </script>
               
</head> """)


// COMMAND ----------

// Création de séquences (Radar Graph) pour les valeurs optimum et means attributes


val rs_1 = Seq(optimum_fractal_dimension, optimum_texture, optimum_perimeter, optimum_area, optimum_smoothness,                                              optimum_concavity, optimum_concave_points, optimum_symmetry,optimum_compactness,optimum_Radius )

val rs_2 = Seq(actual_fractal_dimension, actual_texture, actual_perimeter, actual_area, actual_smoothness, actual_concavity, actual_concave_points, actual_symmetry,actual_compactness,actual_Radius )

val rs_3 = Seq(actual_fractal_dimension_m, actual_texture_m, actual_perimeter_m, actual_area_m, actual_smoothness_m,                                              actual_concavity_m, actual_concave_points_m, actual_symmetry_m, actual_compactness_m, actual_Radius_m )

displayHTML(s""" <head>
               <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
               <div style="width: 1100px;">
               <div id="myDiv_1" style="float:left;width:500px;"></div>
               <div id="myDiv_blank" style="float:left;width:100px;"></div>
               <div id="myDiv_2" style="float:left;width:500px;"></div>
               </div>
               <script>
                      data_1 = [
                       {
                       type: 'scatterpolar',
                       r: [${rs_1(0)}, ${rs_1(1)}, ${rs_1(2)}, ${rs_1(3)}, ${rs_1(4)}, ${rs_1(5)}, ${rs_1(6)}, ${rs_1(7)}, ${rs_1(8)}, ${rs_1(9)}],
                       theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness',  'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Average size of Tumor'
                          
                          },
                          {
                          type: 'scatterpolar',
                          r: [${rs_2(0)}, ${rs_2(1)}, ${rs_2(2)}, ${rs_2(3)}, ${rs_2(4)}, ${rs_2(5)}, ${rs_2(6)}, ${rs_2(7)}, ${rs_2(8)}, ${rs_2(9)}],
                          theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness', 'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Benign Tumor Patient'
                          }
                        ]
                      
                      data_2 = [
                       {
                       type: 'scatterpolar',
                       r: [${rs_1(0)}, ${rs_1(1)}, ${rs_1(2)}, ${rs_1(3)}, ${rs_1(4)}, ${rs_1(5)}, ${rs_1(6)}, ${rs_1(7)}, ${rs_1(8)}, ${rs_1(9)}],
                       theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness',  'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Average size of Tumor'
                          
                          },
                          {
                          type: 'scatterpolar',
                          r: [${rs_3(0)}, ${rs_3(1)}, ${rs_3(2)}, ${rs_3(3)}, ${rs_3(4)}, ${rs_3(5)}, ${rs_3(6)}, ${rs_3(7)}, ${rs_3(8)}, ${rs_3(9)}],
                          theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness', 'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Malignant Cancer Patient'
                          }
                        ]
                        
                      layout_1 = {
                        title : "Analyzing the means across different dimensions for Benign Tumor",
                        "titlefont": {
                                              family : 'Arial', size:14, color:'#000'
                                     }, 
                          polar: {
                        radialaxis: {
      
                              visible: false,
                              range: [0, 1]
                            }
                          }
                        }
                        
                        layout_2 = {
                        
                        title : "Analyzing the means across different dimensions for Malignant Cancer",
                        "titlefont": {
                                              family : 'Arial', size:14, color:'#000'
                                     }, 
                          polar: {
                        radialaxis: {
      
                              visible: false,
                              range: [0, 1]
                            }
                          }
                        }

                        Plotly.plot("myDiv_1", data_1, layout_1)
                        Plotly.plot("myDiv_2", data_2, layout_2)
               </script>
               
</head> """)


// COMMAND ----------


// Définir un tableau des colonnes qui ne sont pas des caractéristiques
val nonFeatureCols = Array("id", "diagnosis", "Int_Diagnostics", "_c32")

// Obtenir toutes les colonnes de DF_features sauf celles définies dans nonFeatureCols
val features = DF_features.columns.diff(nonFeatureCols)



// COMMAND ----------

// Diviser les données en ensembles d'entraînement et de test

// Utiliser la méthode randomSplit pour diviser DF_features en deux ensembles : 80% pour l'entraînement et 20% pour le test
// La graine (seed) est fixée à 12345 pour assurer la reproductibilité des résultats
val Array(training, test) = DF_features.randomSplit(Array(0.8, 0.2), seed = 12345)

// Mettre en cache les données pour accélérer les opérations futures
training.cache()
test.cache()



// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline

// Création de l'assembleur avec les colonnes de caractéristiques et la gestion des valeurs nulles
val assembler = new VectorAssembler()
  .setInputCols(features)
  .setOutputCol("Resultfeatures")
  //.setHandleInvalid("skip") // Ignorer les lignes avec des valeurs nulles

// Création du pipeline pour l'assemblage
val assemblerPipeline = new Pipeline().setStages(Array(assembler))

// Application du pipeline d'assemblage sur les données d'entraînement et de test
val trainingFeatures = assemblerPipeline.fit(training).transform(training)
val testFeatures = assemblerPipeline.fit(test).transform(test)



// COMMAND ----------

// Ajout d'un scaler standard pour mettre à l'échelle les fonctionnalités avant d'appliquer PCA  (normaliser les caractéristiques)
import org.apache.spark.ml.feature.StandardScaler

// Création d'un standard scaler
val scaler = new StandardScaler()
  .setInputCol("Resultfeatures")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false)

// Ajustement du scaler sur les données d'entraînement
val scaler_fit = scaler.fit(trainingFeatures)

// Transformation des données d'entraînement avec le scaler ajusté
val scaler_training = scaler_fit.transform(trainingFeatures)

// Transformation des données de test avec le scaler ajusté
val scaler_test = scaler_fit.transform(testFeatures)


// COMMAND ----------

//Importer les classes nécessaires pour PCA
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors
import scala.collection.mutable

// Inclure PCA
val pca = new PCA()
  .setInputCol("scaledFeatures")
  .setOutputCol("pcaFeatures")
  .setK(2)
  .fit(scaler_training)

//Transformer les données d'entraînement en utilisant le modèle PCA ajusté
val pca_training = pca.transform(scaler_training)

// Transformer les données de test en utilisant le modèle PCA ajusté
val pca_test = pca.transform(scaler_test)

// Sélectionner la colonne 'Int_Diagnostics' à partir des données d'entraînement transformées
val Int_Diagnostics = pca_training.select("Int_Diagnostics")


// COMMAND ----------

//Afficher les résultats du PCA - Comment le PCA se comporte sur l'ensemble de training
val  pca_results= pca.transform(scaler_training).select("pcaFeatures","Int_Diagnostics")


// COMMAND ----------

// Afficher DataFrame pca_results
pca_results.show(false)


// COMMAND ----------

//transforme le vecteur en dataframe 
import org.apache.spark.sql.functions._
import org.apache.spark.ml._

// Création d'un DataFrame à partir d'une séquence  (id, features)
val df = Seq( (1 , linalg.Vectors.dense(1,0) ) ).toDF("id", "features")

// Définition d'une fonction UDF pour convertir VectorUDT en ArrayType
val vecToArray = udf( (xs: linalg.Vector) => xs.toArray)

// Ajout d'une colonne ArrayType au DataFrame en appliquant la fonction UDF aux résultats de PCA
val dfArr = pca_results.withColumn("Features" , vecToArray($"pcaFeatures"))


// COMMAND ----------

// Créer un tableau pour les noms de colonnes
val pca_elements = Array("PCA_1", "PCA_2")

// Créer une expression SQL-like en utilisant le tableau
val sqlExpr = pca_elements.zipWithIndex.map{ case (alias, idx) => col("Features").getItem(idx).as(alias) }

// Extraire les éléments de dfArr et les mettre dans un nouveau DataFrame
val pcaDF = dfArr.select(sqlExpr : _*).toDF

// Sélectionner la colonne "Int_Diagnostics" du DataFrame original dfArr
val newPCADF = dfArr.select("Int_Diagnostics")


// COMMAND ----------

//Creation  d'un identifiant temporaire pour effectuer une opération de jointure

// Importation des types de données Spark SQL
import org.apache.spark.sql.types._

// Création d'un DataFrame Temp_1 avec un index ajouté à partir de newPCADF
val Temp_1 = spark.sqlContext.createDataFrame(
  newPCADF.rdd.zipWithIndex.map {
    case (row, index) => Row.fromSeq(row.toSeq :+ index)
  },


  // Création du schéma pour la colonne index
  StructType(newPCADF.schema.fields :+ StructField("index", LongType, false))
)

// Création d'un DataFrame Temp_2 avec un index ajouté à partir de pcaDF
val Temp_2 = spark.sqlContext.createDataFrame(
  pcaDF.rdd.zipWithIndex.map {
    case (row, index) => Row.fromSeq(row.toSeq :+ index)
  },

  // Création du schéma pour la colonne index
  StructType(pcaDF.schema.fields :+ StructField("index", LongType, false))
)



// COMMAND ----------

// Effectuer une jointure entre Temp_2 et Temp_1 sur la colonne "index" et supprimer la colonne "index" résultante

val final_PCA = Temp_2.join(Temp_1, Seq("index")).drop("index")


// COMMAND ----------

//Achiage  final_PCA
display(final_PCA)


// COMMAND ----------

//Après la mise à l'échelle, PCA fait un travail assez décent en visualisant nos deux clusters cibles (1 pour Malin et 0 pour Benign)

// Bien que la PCA soit très bien capable de différencier les classes, j'ai découvert plus tard que la technique PCA réduisait également certaines caractéristiques importantes, ce qui entraînait une précision réduite. Donc //je n'utiliserai qu'un scaler standard sur l'ensemble de données.

//Construisez les modèles pour prédire le type de tumeur. / XGBoost Classifier, Random Forest Classifier, Light GBM and SVC

// Maintenant que les données ont été préparées, divisons l'ensemble de données d'entraînement en un dataframe d'entraînement et de validation


// COMMAND ----------

// Diviser le DataFrame `pca_training` en ensembles d'entraînement et de validation
// Utiliser un ratio de 80% pour l'entraînement et 20% pour la validation, avec une graine fixe pour la reproductibilité

val Array(trainDF, valDF) = pca_training.randomSplit(Array(0.8, 0.2),seed = 12345)


// COMMAND ----------

// Fonction pour créer une carte de paramètres par défaut pour XGBoost
def get_param_xgb(): mutable.HashMap[String, Any] = {
    // Initialisation d'une carte de paramètres mutable
    val params = new mutable.HashMap[String, Any]()
    // Ajouter les paramètres par défaut pour l'entraînement XGBoost
        params += "eta" -> 0.3  // Taux d'apprentissage
        params += "max_depth" -> 6   // Profondeur maximale 
        params += "gamma" -> 0.0  // Minimum loss reduction required to make a further partition
        params += "colsample_bylevel" -> 1  // Subsample ratio of columns for each split
        params += "objective" -> "binary:logistic"  // Fonction objective pour les tâches de classification binaire
        params += "num_class" -> 2  // Nombre de classes pour la classification
        params += "booster" -> "gbtree"   // Type de modèle de booster
        params += "num_rounds" -> 1   // Nombre de tours d'entraînement
        params += "nWorkers" -> 1   // Nombre de travailleurs pour l'entraînement distribué
    return params    // Retourner la carte de paramètres
}


// COMMAND ----------

//Create XGBoost Classifier, Random Forest Classifier, Light GBM and SVC
val rnf_model = new RandomForestClassifier().setLabelCol("Int_Diagnostics").setFeaturesCol("scaledFeatures")
val lgbm_model = new LightGBMClassifier().setLabelCol("Int_Diagnostics").setFeaturesCol("scaledFeatures")
val svc_model = new LinearSVC().setLabelCol("Int_Diagnostics").setFeaturesCol("scaledFeatures")
val log_model = new LogisticRegression().setLabelCol("Int_Diagnostics").setFeaturesCol("scaledFeatures")


// COMMAND ----------

// Setup the binary classifier evaluator
val evaluator_binary = (new BinaryClassificationEvaluator()
  .setLabelCol("Int_Diagnostics")
  .setRawPredictionCol("Prediction")
  .setMetricName("areaUnderROC"))


// COMMAND ----------

// Fit the model on the training dataset
val fit_rnf = rnf_model.fit(trainDF)

// Transform the training dataset using the fitted model to make predictions
val train_pred_rnf = fit_rnf.transform(trainDF).selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

//Random Forest - Training set
evaluator_binary.evaluate(train_pred_rnf)



// COMMAND ----------

// Afficher les paramètres du modèle Random Forest
println("Printing out the model Parameters:")
println(rnf_model.explainParams)
println("-"*20)


// COMMAND ----------

// Évaluer le modèle sur l'ensemble de validation
val holdout_rnf = fit_rnf
  .transform(valDF)
  .selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

// Évaluer la performance du modèle Random Forest sur l'ensemble de validation en utilisant l'évaluateur binaire

evaluator_binary.evaluate(holdout_rnf)


// COMMAND ----------

// Évaluer le modèle sur l'ensemble de test
val holdout_test_rnf = fit_rnf
  .transform(pca_test)
  .selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

// Évaluer la performance du modèle Random Forest sur l'ensemble de test en utilisant l'évaluateur binaire
evaluator_binary.evaluate(holdout_test_rnf)


// COMMAND ----------

//Light GBM Model

//Ajuster le modèle sur Light GBM Forest
val fit_lgbm = lgbm_model.fit(trainDF)

//Transformer l'ensemble de formation en utilisant le modèle ajusté pour faire des prédictions
val train_pred_lgbm = fit_lgbm.transform(trainDF).selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

// Light GBM - Training set
// Évaluer les performances du modèle Light GBM sur l'ensemble de formation en utilisant l'évaluateur binaire

evaluator_binary.evaluate(train_pred_lgbm)


// COMMAND ----------

// Print the Light GBM Model Parameters
println("Printing out the model Parameters:")
println(lgbm_model.explainParams)
println("-"*20)


// COMMAND ----------

// Évaluer les performances du modèle Light GBM sur l'ensemble de validation
val holdout_lgbm = fit_lgbm
  .transform(valDF)
  .selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_lgbm)

// COMMAND ----------

// Précision sur l'ensemble de test
val holdout_test_lgbm = fit_lgbm
  .transform(pca_test)
  .selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_test_lgbm)


// COMMAND ----------

//Support Vector Classification Model

// Fit the model on SVC
val fit_svc = svc_model.fit(trainDF)
val train_pred_svc = fit_svc.transform(trainDF).selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

//SVC Model - Training set
evaluator_binary.evaluate(train_pred_svc)


// COMMAND ----------

// Affiche  SVC Model Parameters
println("Printing out the model Parameters:")
println(svc_model.explainParams)
println("-"*20)


// COMMAND ----------

// Précision sur l'ensemble de test
val holdout_svc = fit_svc
  .transform(valDF)
  .selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_svc)


// COMMAND ----------

// l'exactitude de l'ensemble de validation
val holdout_test_svc = fit_svc
  .transform(pca_test)
  .selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_test_svc)


// COMMAND ----------

//Logistic Regression Model

// Fit the model on logistic regression model
val fit_log = log_model.fit(trainDF)
val train_pred_log = fit_log.transform(trainDF).selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

//logistic regression Model - Training set
evaluator_binary.evaluate(train_pred_svc)


// COMMAND ----------

// Affiche Logistic regression model Parameters
println("Printing out the model Parameters:")
println(log_model.explainParams)
println("-"*20)


// COMMAND ----------

// l'exactitude de l'ensemble de validation
val holdout_log = fit_log
  .transform(valDF)
  .selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_log)


// COMMAND ----------

// Précision sur l'ensemble de test
val holdout_test_log = fit_log
  .transform(pca_test)
  .selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_test_log)


// COMMAND ----------

//Résumons les résultats de ces 4 modèles, en termes de précision sur l'ensemble de test

// COMMAND ----------

val test_svc = evaluator_binary.evaluate(holdout_test_svc)
val test_rnf = evaluator_binary.evaluate(holdout_test_rnf)
val test_log = evaluator_binary.evaluate(holdout_test_log)
val test_lgbm = evaluator_binary.evaluate(holdout_test_lgbm)

val Xs = Seq(test_rnf,test_svc,test_log,test_lgbm)

displayHTML(s""" <head>
               <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
               
               <div id="tester" style="width:400px;height:400px;"></div>
               
               <script>
                 TESTER = document.getElementById('tester');
                 
                var trace1 = {
                                x: ['Random Forest', 'SVC', 'Logistic', 'Light GBM'],
                                y: [${Xs(0)}, ${Xs(1)}, ${Xs(2)}, ${Xs(3)}],
                                marker:{
                                  color: ['rgba(204,204,204,1)', 'rgba(222,45,38,0.8)', 'rgba(204,204,204,1)','rgba(204,204,204,1)']
                                },
                                type: 'bar'
                              };
             
                              
                              var data = [trace1];

                              var layout = {
                                title: 'Model Comparison Measure : Accuracy',
                                size: 12,
                                   yaxis: {
                                title: 'Accuracy on the test set',
                                range: [.85,.99],                                      
                                  titlefont: {
                                  size: 12,
                                  color: '#7f7f7f'
                                }
                              }
                              };
                 Plotly.newPlot(TESTER, data, layout);
                </script>
               
</head> """)


// COMMAND ----------

// SVC model - Training Set

val lp_svc = train_pred_svc.select("Prediction", "Int_Diagnostics")

//Total Records
val counttotal = train_pred_svc.count()

// True Positives + False Negatives
val correct = lp_svc.filter($"Int_Diagnostics" === $"Prediction").count()

// True Negatives + False Positives
val wrong = lp_svc.filter(not($"Int_Diagnostics" === $"Prediction")).count()

//True Positive
val truep = lp_svc.filter($"Prediction" === 1.0).filter($"Int_Diagnostics" === $"Prediction").count()

//True Negative
val falseN = lp_svc.filter($"Prediction" === 0.0).filter($"Int_Diagnostics" === $"Prediction").count()

// False Negative
val falseP = lp_svc.filter($"Prediction" === 0.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//False Positive
val truen = lp_svc.filter($"Prediction" === 1.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//Precision
val Precision = truep.toDouble / (truep.toDouble + falseP.toDouble)

//Recall
val Recall = truep.toDouble / (truep.toDouble + falseN.toDouble)


// COMMAND ----------

//Obtenez l'aire sous la courbe
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
val predictionLabels = train_pred_svc.select("Prediction", "Int_Diagnostics").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
val binMetrics = new BinaryClassificationMetrics(predictionLabels)

val roc_log_train = binMetrics.areaUnderROC


// COMMAND ----------

// SVC Model - validation Set

val lp_svc = holdout_svc.select("Prediction", "Int_Diagnostics")

//Total Records
val counttotal = holdout_svc.count()

// True Positives + False Negatives
val correct = lp_svc.filter($"Int_Diagnostics" === $"Prediction").count()

// True Negatives + False Positives
val wrong = lp_svc.filter(not($"Int_Diagnostics" === $"Prediction")).count()

//True Positive
val truep = lp_svc.filter($"Prediction" === 1.0).filter($"Int_Diagnostics" === $"Prediction").count()

//True Negative
val falseN = lp_svc.filter($"Prediction" === 0.0).filter($"Int_Diagnostics" === $"Prediction").count()

// False Negative
val falseP = lp_svc.filter($"Prediction" === 0.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//False Positive
val truen = lp_svc.filter($"Prediction" === 1.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//Precision
val Precision = truep.toDouble / (truep.toDouble + falseP.toDouble)

//Recall
val Recall = truep.toDouble / (truep.toDouble + falseN.toDouble)


// COMMAND ----------

// ROC for Vaidation Set - Logistic Regression

val predictionLabels = holdout_svc.select("Prediction", "Int_Diagnostics").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
val binMetrics = new BinaryClassificationMetrics(predictionLabels)

val roc_log_train = binMetrics.areaUnderROC


// COMMAND ----------

// SVC Model - Test Set

val lp_log = holdout_test_svc.select("Prediction", "Int_Diagnostics")

//Total Records
val counttotal = holdout_test_svc.count()

// True Positives + False Negatives
val correct = lp_log.filter($"Int_Diagnostics" === $"Prediction").count()

// True Negatives + False Positives
val wrong = lp_log.filter(not($"Int_Diagnostics" === $"Prediction")).count()

//True Positive
val truep = lp_log.filter($"Prediction" === 1.0).filter($"Int_Diagnostics" === $"Prediction").count()

//True Negative
val falseN = lp_log.filter($"Prediction" === 0.0).filter($"Int_Diagnostics" === $"Prediction").count()

// False Negative
val falseP = lp_log.filter($"Prediction" === 0.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//False Positive
val truen = lp_log.filter($"Prediction" === 1.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//Precision
val Precision = truep.toDouble / (truep.toDouble + falseP.toDouble)

//Recall
val Recall = truep.toDouble / (truep.toDouble + falseN.toDouble)


// COMMAND ----------

// ROC for Vaidation Set - SVC Classification Model

val predictionLabels = holdout_test_svc.select("Prediction", "Int_Diagnostics").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
val binMetrics = new BinaryClassificationMetrics(predictionLabels)

val roc_log_train = binMetrics.areaUnderROC

