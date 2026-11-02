import difflib
import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


args = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET_SILVER", "BUCKET_GOLD"])
JOB_NAME = args["JOB_NAME"]
BUCKET_SILVER = args["BUCKET_SILVER"]
BUCKET_GOLD = args["BUCKET_GOLD"]

ALUNOS_PATH = "alfabetizacao_inep/aluno"
MUNICIPIOS_PATH = "alfabetizacao_inep/municipio"
ESTADOS_PATH = "alfabetizacao_inep/estado"
CENSO_PATH = "censo_escolar/censo_escolar_municipal"
CENSO_2022_PATH = "censo_2022/censo_2022.csv"
FUNDEB_PATH = "fundeb/fundeb_anual"
IDHM_PATH = "idhm/idhm_uf"
GOLD_PATH = "alfabetizacao/base_alunos"
ACENTUADOS = "áàâãéêíóôõúç"
SEM_ACENTO = "aaaaeeiooouc"
UF_BY_CODE = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR",
    42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF",
}

FINAL_COLUMNS = [
    "year", "state_code", "student_id", "grade", "school_id",
    "education_network_type", "city_code", "student_exam_weight",
    "student_exam_score", "student_is_literate", "city_name",
    "city_percentage_literate_students", "city_average_exam_score",
    "state_abbreviation", "state_percentage_literate_students",
    "state_average_exam_score", "city_name_norm", "households", "area",
    "literacy_rate", "median_age", "sex_ratio", "aging_index",
    "population_total", "pct_white", "pct_black", "pct_yellow",
    "pct_brown", "pct_indigenous", "fundeb_amount", "school_count",
    "total_enrollments", "total_teachers", "total_classes",
    "pct_rural_schools", "pct_internet_schools", "pct_library_schools",
    "pct_reading_room_schools", "pct_reading_space_schools",
    "pct_full_time_enrollments", "students_per_teacher",
    "students_per_class", "pct_enrollments_zero_class_schools",
    "null_teacher_schools_count", "year_idhm", "year_join", "hdi",
    "hdi_life_expectancy", "hdi_education", "hdi_income", "gini_index",
    "income_per_capita_mean", "age_lag_2y_basic_rate",
]


def read_silver(spark, bucket_silver: str, path: str) -> DataFrame:
    """Lê uma tabela Silver em Parquet."""

    return spark.read.parquet(f"{bucket_silver}/{path}")


def read_silver_csv(spark, bucket_silver: str, path: str) -> DataFrame:
    """Lê a tabela CSV do Censo 2022 produzida pela Silver."""

    return (
        spark.read
        .option("header", True)
        .option("sep", ";")
        .option("encoding", "UTF-8")
        .csv(f"{bucket_silver}/{path}")
    )


def chave_municipio(coluna: str):
    """Normaliza município em minúsculas e sem acentos, como no notebook."""

    return F.translate(F.lower(F.trim(F.col(coluna))), ACENTUADOS, SEM_ACENTO)


def optional_numeric(dataframe: DataFrame, column: str):
    """Lê uma coluna numérica quando a Silver a disponibiliza; senão, cria nulo tipado."""

    if column in dataframe.columns:
        return F.col(column).cast("double").alias(column)
    return F.lit(None).cast("double").alias(column)


def normalize_students(alunos: DataFrame) -> DataFrame:
    """Converte a tabela Silver de alunos para o schema usado no notebook."""

    return alunos.select(
        F.col("ano").cast("int").alias("year"),
        F.col("co_uf").cast("int").alias("state_code"),
        F.col("id_aluno").alias("student_id"),
        F.col("tp_serie").cast("int").alias("grade"),
        F.col("id_escola").alias("school_id"),
        F.col("tp_dependencia").cast("int").alias("education_network_type"),
        F.col("co_municipio").cast("int").alias("city_code"),
        F.col("no_municipio").alias("student_city_name"),
        F.col("in_presenca_lp").cast("int").alias("student_is_present_exam"),
        F.col("in_preenchimento_lp").cast("int").alias("student_took_exam"),
        F.col("vl_peso_aluno_lp").cast("double").alias("student_exam_weight"),
        F.col("vl_proficiencia_lp").cast("double").alias("student_exam_score"),
        F.col("in_alfabetizado").cast("int").alias("student_is_literate"),
    )


def normalize_municipal_exam(municipios: DataFrame) -> DataFrame:
    """Converte a dimensão municipal de alfabetização para o schema do notebook."""

    return (
        municipios.select(
            F.col("ano").cast("int").alias("year"),
            F.col("co_uf").cast("int").alias("state_code"),
            F.col("co_municipio").cast("int").alias("city_code"),
            F.col("no_municipio").alias("city_name"),
            F.col("tp_serie").cast("int").alias("grade"),
            F.col("id_tipo_rede").cast("int").alias("education_network_type"),
            F.col("pc_aluno_alfabetizado").cast("double").alias("city_percentage_literate_students"),
            F.col("vl_media_lp").cast("double").alias("city_average_exam_score"),
        )
        .dropDuplicates(["year", "state_code", "grade", "education_network_type", "city_code"])
    )


def normalize_state_exam(estados: DataFrame) -> DataFrame:
    """Converte a dimensão estadual de alfabetização para o schema do notebook."""

    return (
        estados.select(
            F.col("ano").cast("int").alias("year"),
            F.col("co_uf").cast("int").alias("state_code"),
            F.col("sg_uf").alias("state_abbreviation"),
            F.col("tp_serie").cast("int").alias("grade"),
            F.col("id_tipo_rede").cast("int").alias("education_network_type"),
            F.col("pc_aluno_alfabetizado").cast("double").alias("state_percentage_literate_students"),
            F.col("vl_media_lp").cast("double").alias("state_average_exam_score"),
        )
        .dropDuplicates(["year", "state_code", "grade", "education_network_type"])
    )


def align_fundeb_names(fundeb: DataFrame, base: DataFrame) -> DataFrame:
    """Aplica a mesma correção aproximada de nomes usada no notebook."""

    valid_names = {
        (row["state_abbreviation"], row["city_name_norm"])
        for row in base.select("state_abbreviation", "city_name_norm").dropna().distinct().collect()
    }
    fundeb_names = [
        (row["state_abbreviation"], row["fundeb_city_name_norm"])
        for row in fundeb.select("state_abbreviation", "fundeb_city_name_norm").dropna().distinct().collect()
    ]

    corrections = []
    for state, fundeb_name in fundeb_names:
        if (state, fundeb_name) in valid_names:
            continue
        candidates = [name for uf, name in valid_names if uf == state]
        match = difflib.get_close_matches(fundeb_name, candidates, n=1, cutoff=0.8)
        if match:
            corrections.append((state, fundeb_name, match[0]))

    if not corrections:
        return fundeb.withColumnRenamed("fundeb_city_name_norm", "city_name_norm")

    mapping = base.sparkSession.createDataFrame(
        corrections,
        ["state_abbreviation", "fundeb_city_name_norm", "city_name_norm"],
    )
    return (
        fundeb.join(mapping, ["state_abbreviation", "fundeb_city_name_norm"], "left")
        .withColumn("city_name_norm", F.coalesce(F.col("city_name_norm"), F.col("fundeb_city_name_norm")))
        .drop("fundeb_city_name_norm")
    )


def build_gold(
    alunos: DataFrame,
    municipios: DataFrame,
    estados: DataFrame,
    censo: DataFrame,
    censo_2022: DataFrame,
    fundeb: DataFrame,
    idhm: DataFrame,
) -> DataFrame:
    """Reproduz os merges, filtros e o schema final do ``eda.ipynb``."""

    base = normalize_students(alunos).filter(F.col("education_network_type").isin(2, 3))

    # O notebook remove city_name dos alunos antes do merge e usa o nome da dimensão municipal.
    base = base.join(
        F.broadcast(normalize_municipal_exam(municipios)),
        ["year", "state_code", "grade", "education_network_type", "city_code"],
        "left",
    )
    base = base.join(
        F.broadcast(normalize_state_exam(estados)),
        ["year", "state_code", "grade", "education_network_type"],
        "left",
    )

    # Fallbacks usados no notebook para joins sem correspondência.
    uf_map = F.create_map(*[item for pair in UF_BY_CODE.items() for item in (F.lit(pair[0]), F.lit(pair[1]))])
    base = (
        base.withColumn("state_abbreviation", F.coalesce(F.col("state_abbreviation"), uf_map[F.col("state_code")]))
        .withColumn("city_name", F.coalesce(F.col("city_name"), F.col("student_city_name")))
    )

    # Imputação estadual e fallback municipal -> estadual, como no EDA.
    state_window = Window.partitionBy("year", "grade")
    base = (
        base.withColumn(
            "state_percentage_literate_students",
            F.coalesce(
                F.col("state_percentage_literate_students"),
                F.avg("state_percentage_literate_students").over(state_window),
            ),
        )
        .withColumn(
            "state_average_exam_score",
            F.coalesce(
                F.col("state_average_exam_score"),
                F.avg("state_average_exam_score").over(state_window),
            ),
        )
        .withColumn(
            "city_percentage_literate_students",
            F.coalesce(F.col("city_percentage_literate_students"), F.col("state_percentage_literate_students")),
        )
        .withColumn(
            "city_average_exam_score",
            F.coalesce(F.col("city_average_exam_score"), F.col("state_average_exam_score")),
        )
    )

    # Reclassificação feita no notebook antes do filtro de presença.
    base = base.withColumn(
        "student_is_present_exam",
        F.when(
            F.col("student_exam_score").isNull() & (F.col("student_is_present_exam") == 1),
            0,
        ).otherwise(F.col("student_is_present_exam")),
    )

    # Censo 2022: mantém população total e transforma população racial em percentual.
    racial_groups = ["white", "black", "yellow", "brown", "indigenous"]
    censo_2022_ref = censo_2022.select(
        F.col("city_code").cast("int").alias("city_code"),
        optional_numeric(censo_2022, "households"),
        optional_numeric(censo_2022, "area"),
        optional_numeric(censo_2022, "literacy_rate"),
        optional_numeric(censo_2022, "median_age"),
        optional_numeric(censo_2022, "sex_ratio"),
        optional_numeric(censo_2022, "aging_index"),
        F.col("population_total").cast("double").alias("population_total"),
        *[
            (F.col(f"population_{group}") / F.col("population_total")).alias(f"pct_{group}")
            for group in racial_groups
        ],
    ).dropDuplicates(["city_code"])
    base = base.join(F.broadcast(censo_2022_ref), ["city_code"], "left")

    # FUNDEB usa o mesmo ano e a chave textual normalizada.
    fundeb_ref = (
        fundeb.select(
            F.col("year").cast("int").alias("year"),
            chave_municipio("city_name").alias("fundeb_city_name_norm"),
            F.col("state_abbreviation"),
            F.col("value").cast("double").alias("fundeb_amount"),
        )
        .dropDuplicates(["year", "fundeb_city_name_norm", "state_abbreviation"])
    )
    base = base.withColumn("city_name_norm", chave_municipio("city_name"))
    fundeb_ref = align_fundeb_names(fundeb_ref, base)
    base = base.join(
        F.broadcast(fundeb_ref),
        ["year", "city_name_norm", "state_abbreviation"],
        "left",
    )

    # Censo Escolar usa o mesmo ano do aluno.
    censo_ref = censo.select(
        F.col("ano").cast("int").alias("year"),
        F.col("id_municipio").cast("int").alias("city_code"),
        F.col("qtd_escolas").cast("double").alias("school_count"),
        F.col("total_matriculas").cast("double").alias("total_enrollments"),
        F.col("total_docentes").cast("double").alias("total_teachers"),
        F.col("total_turmas").cast("double").alias("total_classes"),
        F.col("pct_escolas_rurais").cast("double").alias("pct_rural_schools"),
        F.col("pct_escolas_internet").cast("double").alias("pct_internet_schools"),
        F.col("pct_escolas_biblioteca").cast("double").alias("pct_library_schools"),
        F.col("pct_escolas_sala_leitura").cast("double").alias("pct_reading_room_schools"),
        F.col("pct_escolas_espaco_leitura").cast("double").alias("pct_reading_space_schools"),
        F.col("pct_matriculas_integral").cast("double").alias("pct_full_time_enrollments"),
        F.col("alunos_por_docente").cast("double").alias("students_per_teacher"),
        F.col("alunos_por_turma").cast("double").alias("students_per_class"),
        F.col("pct_matriculas_escolas_turmas_zero").cast("double").alias("pct_enrollments_zero_class_schools"),
        F.col("qtd_escolas_docentes_nulos").cast("double").alias("null_teacher_schools_count"),
    ).dropDuplicates(["city_code", "year"])
    base = base.join(F.broadcast(censo_ref), ["city_code", "year"], "left")
    base = base.withColumn("students_per_teacher", F.coalesce(F.col("students_per_teacher"), F.lit(0.0)))

    # IDHM: year_join = ano do IDHM + 1, mantendo year_idhm e year_join na saída.
    idhm_ref = idhm.select(
        F.col("year").cast("int").alias("year_idhm"),
        F.col("state_code").cast("int").alias("idhm_state_code"),
        (F.col("year") + 1).cast("int").alias("year_join"),
        F.col("hdi").cast("double").alias("hdi"),
        F.col("hdi_life_expectancy").cast("double").alias("hdi_life_expectancy"),
        F.col("hdi_education").cast("double").alias("hdi_education"),
        F.col("hdi_income").cast("double").alias("hdi_income"),
        F.col("gini_index").cast("double").alias("gini_index"),
        F.col("income_per_capita_mean").cast("double").alias("income_per_capita_mean"),
        F.col("age_lag_2y_basic_rate").cast("double").alias("age_lag_2y_basic_rate"),
    ).dropDuplicates(["idhm_state_code", "year_join"])
    base = base.join(
        F.broadcast(idhm_ref),
        (base["state_code"] == idhm_ref["idhm_state_code"])
        & (base["year"] == idhm_ref["year_join"]),
        "left",
    ).drop("idhm_state_code")

    # Mesmo tratamento final do notebook.
    base = base.filter(F.col("student_is_present_exam") == 1)
    base = base.filter(F.col("population_total").isNotNull())
    median_fundeb = F.percentile_approx("fundeb_amount", 0.5).over(
        Window.partitionBy("state_abbreviation", "year")
    )
    base = base.withColumn("fundeb_amount", F.coalesce(F.col("fundeb_amount"), median_fundeb))
    return base.drop("student_city_name", "student_is_present_exam", "student_took_exam").select(FINAL_COLUMNS)


def load(dataframe: DataFrame, bucket_gold: str, path: str) -> str:
    """Grava a base consolidada em Parquet na Gold, particionada por ano."""

    destino = f"{bucket_gold}/{path}"
    dataframe.write.mode("overwrite").partitionBy("year").parquet(destino)
    return destino


sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(JOB_NAME, args)

alunos = read_silver(spark, BUCKET_SILVER, ALUNOS_PATH)
municipios = read_silver(spark, BUCKET_SILVER, MUNICIPIOS_PATH)
estados = read_silver(spark, BUCKET_SILVER, ESTADOS_PATH)
censo = read_silver(spark, BUCKET_SILVER, CENSO_PATH)
censo_2022 = read_silver_csv(spark, BUCKET_SILVER, CENSO_2022_PATH)
fundeb = read_silver(spark, BUCKET_SILVER, FUNDEB_PATH)
idhm = read_silver(spark, BUCKET_SILVER, IDHM_PATH)

gold = build_gold(alunos, municipios, estados, censo, censo_2022, fundeb, idhm)
load_path = load(gold, BUCKET_GOLD, GOLD_PATH)
print(
    f"Base Gold gravada em {load_path}: {gold.count()} registros e "
    f"{len(gold.columns)} colunas",
    flush=True,
)

job.commit()
