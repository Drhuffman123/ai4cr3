require "yaml"

class FooBar
  include YAML::Serializable

  property a = [1,2,3]
  property b = Time.local
  property c = [1.23,2.34]

  # def initialize(a,b,c)
  #   @a = a
  #   @b = b
  #   @c = c
  # end
end

afoobar = FooBar.new([1,2,3],Time.local,[1.23,2.34])
puts afoobar.to_s
# puts afoobar.to_yaml
puts YAML.dump(afoobar)
foo = FooBar.from_yaml(%({"a": [1,2,3], "b": "lat", "c": [34.5]})
puts foo
# yaml = YAML.dump({hello: "world"}) 
# yaml = afoobar.to_yaml
# puts yaml
