# ai4cr3

Version: 0.0.10

[![Crystal CI](https://github.com/drhuffman123/ai4cr3/actions/workflows/crystal.yml/badge.svg)](https://github.com/drhuffman12/ai4cr3/actions/workflows/crystal.yml)

## Installation

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     ai4cr3:
       github: drhuffman12/ai4cr3
   ```

2. Run `shards install`

## Usage

```crystal
require "ai4cr3"
```

TODO: Write usage instructions here

## Development

TODO: Write development instructions here

## Contributing
* RESET via:
  * Reset the a fresh up to date repo copy (or a fork):
    * `gh repo clone Drhuffman123/ai4cr3`
    * `cd Drhuffman123/ai4cr3`
    * `git reset --hard; git checkout main; git pull origin main`
  * Add a new branch:
    * `git checkout -b YOUR_BREANCH_NAME`
  * ADD your CHANGES (`git add my-new-files`)
  * (Please) correctly update the VERSION!!! (Edit/Syncup `src/ai4cr3/about.cr` and `shard.yml` and in the notes above, under `SHOULD MATCH`)
  * Add your changes and check them with:
    * `crystal tool format`
    * `crystal spec --error-trace`
  * VERIFY your `Changes to be committed` (`git status`)
  * Update your version in these files and git add them:
    * `git add README.md; git add ai4cr3`
  * Commit your changes and git push them:
    * `git commit -am YOUR_COMMIT_MESSAGE`
    * `git push origin YOUR_BREANCH_NAME`
  * Verify the whole thing locally w/ act:
    * `act -r`

## Contributors

This is a re-port of ai4r from https://github.com/SergioFierens/ai4r to Crystal
- [Daniel Huffman](https://github.com/drhuffman12) - creator and maintainer
