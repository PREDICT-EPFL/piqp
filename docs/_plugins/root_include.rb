class RootInclude < Liquid::Tag
  def initialize(_tag_name, markup, _parse_context)
    super
    @markup = markup.strip
  end

  def render(context)
    expanded_path = Liquid::Template.parse(@markup).render(context)
    root_path = File.expand_path(context.registers[:site].config['source'])
    final_path = File.join(root_path, expanded_path)
    read_file(final_path, context)
  end

  def read_file(path, context)
    file_read_opts = context.registers[:site].file_read_opts
    File.read(path, **file_read_opts)
  end
end

Liquid::Template.register_tag('root_include', RootInclude)