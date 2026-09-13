# ============================================
# TITANIC DATASET - SURVIVAL ANALYSIS
# ============================================
#
# Objective:
# Analyze the Titanic dataset and identify
# factors associated with passenger survival.
#
# Dataset:
# Titanic Dataset - 891 passengers, 12 columns
#
# Main areas analyzed:
# - Overall survival
# - Gender
# - Passenger class
# - Age groups
# - Gender and passenger class
# - Family size
# ============================================


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================
# 1. Load Dataset
# ===================================

file_path = r"D:\Beeskilled internship DataScience inpython\Week4_Project\Titanic-Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

# ===================================
# 2. Basic Cleaning
# ===================================
df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ===================================
# 3. Overall Survival Analysis
# ===================================

survival_counts = df["Survived"].value_counts()

print("\nSurvival Count:")
print(survival_counts)

survival_rate = df["Survived"].mean() * 100

print(f"\nOverall Survival Rate: {survival_rate:.2f}%")


plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Survived")

plt.title("Titanic Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"]
)

plt.tight_layout()
plt.show()

# ===================================
# 4. Survival by Gender
# ===================================

gender_survival = df.groupby("Sex")["Survived"].mean() * 100

print("\nSurvival Rate by Gender:")
print(gender_survival)


plt.figure(figsize=(7, 5))

ax = sns.barplot(
    data=df,
    x="Sex",
    y="Survived"
)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")

ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(
    ["0%", "20%", "40%", "60%", "80%"]
)
for i, (idx, value) in enumerate(gender_survival.items()):
    ax.text(
        idx,
        value / 200,
        f"{value:.1f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()


# ===================================
# 5. Survival by Passenger Class
# ===================================

class_survival = df.groupby("Pclass")["Survived"].mean() * 100

print("\nSurvival Rate by Passenger Class:")
print(class_survival)


plt.figure(figsize=(7, 5))

ax = sns.barplot(
    data=df,
    x="Pclass",
    y="Survived"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")

ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(
    ["0%", "20%", "40%", "60%", "80%"]
)

for i, (idx, value) in enumerate(class_survival.items()):
    ax.text(
        i,
        value / 200,
        f"{value:.1f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()


# ===================================
# 6. Age Group Analysis
# ===================================

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 17, 35, 59, 100],
    labels=[
        "Child (0-12)",
        "Teenager (13-17)",
        "Young Adult (18-35)",
        "Adult (36-59)",
        "Senior (60+)"
    ]
)

age_survival = (
    df.groupby(
        "AgeGroup",
        observed=True
    )["Survived"].mean() * 100
)

print("\nSurvival Rate by Age Group:")
print(age_survival)


plt.figure(figsize=(9, 5))

ax = sns.barplot(
    x=age_survival.index,
    y=age_survival.values
)

plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")

plt.xticks(rotation=20)

# Add percentage labels
for i, value in enumerate(age_survival.values):
    ax.text(
        i,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()


# ===================================
# 7. Survival by Gender and Class
# ===================================

gender_class_survival = (
    df.groupby(
        ["Pclass", "Sex"]
    )["Survived"].mean() * 100
)

print("\nSurvival Rate by Passenger Class and Gender:")
print(gender_class_survival)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=df,
    x="Pclass",
    y="Survived",
    hue="Sex"
)

plt.title("Survival Rate by Passenger Class and Gender")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")

ax.set_yticks(
    [0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

ax.set_yticklabels(
    ["0%", "20%", "40%", "60%", "80%", "100%"]
)

plt.legend(title="Gender")

plt.tight_layout()
plt.show()

# ===================================
# 8. Family Size Analysis
# ===================================

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df["FamilyCategory"] = pd.cut(
    df["FamilySize"],
    bins=[0, 1, 4, 20],
    labels=[
        "Alone",
        "Small Family",
        "Large Family"
    ]
)

family_survival = (
    df.groupby(
        "FamilyCategory",
        observed=True
    )["Survived"].mean() * 100
)

print("\nSurvival Rate by Family Category:")
print(family_survival)


plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=family_survival.index,
    y=family_survival.values
)

plt.title("Survival Rate by Family Category")
plt.xlabel("Family Category")
plt.ylabel("Survival Rate (%)")

plt.ylim(0, 70)

plt.yticks(
    range(0, 71, 10),
    [f"{x}%" for x in range(0, 71, 10)]
)

for i, value in enumerate(family_survival.values):
    ax.text(
        i,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()

# ===================================
# 9. Key Findings and Observations
# ===================================

print("""
KEY FINDINGS:

1. Overall Survival:
   Out of 891 passengers, 342 survived and 549 did not.
   The overall survival rate was 38.38%.

2. Gender:
   Female passengers had a survival rate of 74.20%,
   while male passengers had a survival rate of 18.89%.

3. Passenger Class:
   First-class passengers had the highest survival rate
   at 62.96%, followed by second class at 47.28% and
   third class at 24.24%.

4. Gender and Class:
   Female passengers had higher survival rates than male
   passengers in every passenger class. First-class females
   had the highest survival rate at 96.81%, while third-class
   males had the lowest at 13.54%.

5. Age:
   Survival rates varied across age groups. Children had a
   survival rate of 57.97%, while seniors had a survival
   rate of 26.92%.

6. Family Size:
   Passengers travelling with a small family had the highest
   survival rate at 57.88%. Large-family passengers had the
   lowest survival rate at 16.13%.

CONCLUSION:

The analysis shows that survival on the Titanic was strongly
associated with passenger gender and class. Female passengers
and higher-class passengers generally had substantially higher
survival rates. Age and family size also showed differences
in survival outcomes.
""")