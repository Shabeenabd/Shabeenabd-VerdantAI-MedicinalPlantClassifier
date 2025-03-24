<?php 
if(!class_exists('element_gva_text')):
   class element_gva_text{
      public function render_form(){
         $option_1 = array(
            ''   => '--Default--',
            '14'   => '14',
            '16'   => '16',
            '18'   => '18',
            '20'   => '20',
            '22'   => '22',
            '24'   => '24',
            '26'   => '26',
            '28'   => '28',
            '30'   => '30',
            '32'   => '32',
            '34'   => '34',
            '36'   => '36',
            '38'   => '38',
            '40'   => '40',
            '42'   => '42',
            '44'   => '44',
            '46'   => '46',
            '48'   => '48',
            '50'   => '50',
            '52'   => '52',
            '54'   => '54',
            '56'   => '56',
            '58'   => '58',
            '60'   => '60',
            '70'   => '70',
            '80'   => '80',
            '90'   => '90',
            '100'  => '100'
         );
         $fields = array(
            'type' => 'gsc_text',
            'title' => t('Custom Text'),
            'fields' => array(
               array(
                  'id'     => 'title',
                  'type'      => 'text',
                  'title'  => t('Title'),
                   'admin'     => true
               ),
               array(
                  'id'           => 'content',
                  'type'         => 'textarea',
                  'title'        => t('Column content'),
                  'desc'         => t('Shortcodes and HTML tags allowed.'),
                  'shortcodes'   => 'on'
               ),
               array(
                  'id'        => 'title_font_size',
                  'type'      => 'select',
                  'title'     => t('Title Font Size'),
                  'options'   => $option_1,
                  'default'   => ''
               ),
               array(
                  'id'        => 'style',
                  'type'      => 'select',
                  'title'     => t('Style display'),
                  'options'   => array(
                        ''   => 'Default',
                        'text-size-medium'  => 'Text medium',
                  )
               ),
               array(
                  'id'        => 'margin',
                  'type'      => 'select',
                  'title'     => t('Margin Bottom'),
                  'options'   => array(
                     'box-margin-0'       => t('Remove Margin Bottom'), 
                     'box-margin-small'   => t('Margin Bottom Small'), 
                     'box-margin-medium'  => t('Margin Bottom Medium'),
                     'box-margin-large'   => t('Margin Bottom Large'),
                  ),
                  'default'   => 'box-margin-0'
               ), 
               array(
                  'id'     => 'animate',
                  'type'      => 'select',
                  'title'  => ('Animation'),
                  'desc'  => t('Entrance animation for element'),
                  'options'   => gavias_content_builder_animate(),
                  'class'     => 'width-1-2'
               ),
               array(
                  'id'        => 'animate_delay',
                  'type'      => 'select',
                  'title'     => t('Animation Delay'),
                  'options'   => gavias_content_builder_delay_aos(),
                  'desc'      => '0 = default',
                  'class'     => 'width-1-2'
               ), 
               array(
                  'id'        => 'el_class',
                  'type'      => 'text',
                  'title'     => t('Extra class name'),
                  'desc'      => t('Style particular content element differently - add a class name and refer to it in custom CSS.'),
               ),   
            ),                                     
         );
         return $fields;
      }

      public static function render_content( $attr = array(), $content = '' ){
         extract(gavias_merge_atts(array(
            'title'              => '',
            'content'            => '',
            'title_font_size'    => '',
            'style'              => '',
            'margin'             => '',
            'el_class'           => '',
            'animate'            => '',
            'animate_delay'      => '',
         ), $attr));
         $el_class .= ' ' . $style;
         $array_style = array();
         if( $title_font_size){           
            $array_style[] = 'font-size:' . $title_font_size . 'px';
         }

         $text_style  = implode('; ', $array_style );
         if($animate) $el_class .= ' wow ' . $animate;
         
         $ouput = '';
         $ouput .= '<div style=" '.$text_style.'" class="custom-text '.$el_class.' '.$margin.'" '. gavias_content_builder_print_animate_wow('', $animate_delay) .'>';
         $ouput .=  $content;
         $ouput .= '</div>';
         return $ouput;
      }

   }
 endif;  



