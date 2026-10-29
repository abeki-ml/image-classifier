from pathlib import Path

data_dir = Path("data/PetImages")

cat_images = list((data_dir / "Cat").glob("*jpg"))
dog_images = list((data_dir / "Dog").glob("*.jpg"))

print("Cat images:", len(cat_images))
print("Dog images:", len(dog_images))
