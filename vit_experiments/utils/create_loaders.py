from torch.utils.data import DataLoader, TensorDataset

def create_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test):
  # Cria datasets
  train_dataset = TensorDataset(X_train, Y_train)
  val_dataset = TensorDataset(X_val, Y_val)
  dataset_test = TensorDataset(X_test, Y_test)

  # Cria DataLoaders
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
  test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

  return train_loader, val_loader, test_loader
