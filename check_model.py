import joblib

model = joblib.load("shg_loan_model.joblib")

print("MODEL TYPE:")
print(type(model))

print("\nMODEL STEPS:")
print(model.named_steps)

preprocessor = model.named_steps["preprocessing"]

print("\nPREPROCESSOR:")
print(preprocessor)

encoder = preprocessor.named_transformers_["cat"]

print("\nENCODER CATEGORIES:")
for column, categories in zip(
    preprocessor.transformers_[0][2],
    encoder.categories_
):
    print("\nColumn:", column)
    print("Categories:", categories)
    print("Category types:", [type(x) for x in categories])