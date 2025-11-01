from src.resampling import apply_smote, print_class_balance
X_res, y_res = apply_smote(X_orig, y_orig)
print_class_balance(y_orig, "Original")
print_class_balance(y_res, "Resampled")
