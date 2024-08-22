{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   {% for item in attributes %}
   {%- if not item in ['loc', 'iloc', 'loc_annot', 'iloc_annot'] %}
   .. autoattribute:: {{ name }}.{{ item }}
   {%- endif %}

   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
   {% for item in all_methods %}
   {%- if not item.startswith('_') or item in ['__len__', '__getitem__'] %}
      ~{{ name }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
   {% for item in attributes %}
   {%- if item in ['loc', 'iloc', 'loc_annot', 'iloc_annot'] %}
      ~{{ name }}.{{ item }}
   {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
