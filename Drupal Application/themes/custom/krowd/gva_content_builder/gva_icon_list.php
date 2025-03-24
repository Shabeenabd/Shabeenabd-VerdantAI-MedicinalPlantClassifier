<?php 
if(!class_exists('element_gva_icon_list')){
   class element_gva_icon_list{
      public function render_form(){
         $fields = array(
            'type'   => 'gsc_icon_list',
            'title'  => t('Icon List'), 
            'fields' => array(      
               array(
                  'id'        => 'title',
                  'type'      => 'text',
                  'title'     => t('Title for admin'),
                  'admin'     => true
               ),
               array(
                  'id'        => 'animate',
                  'type'      => 'select',
                  'title'     => t('Animation'),
                  'desc'      => t('Entrance animation for element'),
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
         for($i=1; $i<=8; $i++){
            $fields['fields'][] = array(
               'id'     => "info_{$i}",
               'type'   => 'info',
               'desc'   => "Information for item {$i}"
            );

            $fields['fields'][] = array(
               'id'        => "icon_{$i}",
               'type'      => 'text',
               'title'     => t("Icon class {$i}"),
               'default'   => '',
               'desc'     => t('Use class icon font <a target="_blank" href="http://fontawesome.io/icons/">Icon Awesome</a> or <a target="_blank" href="http://gaviasthemes.com/icons/ionicon/">Custom icon</a>'),
            );
            $fields['fields'][] = array(
               'id'        => "title_{$i}",
               'type'      => 'text',
               'title'     => t("Title {$i}")
            );
         }
         return $fields;
      }

      public static function render_content( $attr = array(), $content = '' ){
         global $base_url;
         $default = array(
            'title'           => '',
            'el_class'        => '',
            'animate'         => '',
            'animate_delay'   => ''
         );  

            
         for($i=1; $i<=8; $i++){
            $default["title_{$i}"] = '';
            $default["icon_{$i}"] = '';
         }   
         extract(gavias_merge_atts($default, $attr)); 
         
         $_id = gavias_content_builder_makeid();
         
         if($animate) $el_class .= ' wow ' . $animate;
         
         ob_start() ?>
         <div class="gsc-icon-list <?php print $el_class ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>
            <ul class="list-unstyled">
               <?php for($i=1; $i<=8; $i++){ ?>
                  <?php 
                     $title = "title_{$i}";
                     $icon_list = "icon_{$i}";
                  ?>
                  <?php if($$icon_list || $$title){ ?>
                  <li>
                     <?php if($$icon_list){ ?>
                        <i class="icon <?php print $$icon_list ?>"></i>
                     <?php } ?>
                     <?php if($$title){ ?>
                     <span><?php print $$title ?> </span>
                     <?php } ?>
                  </li>
                  <?php } ?>
               <?php } ?>
            </ul>
               
         </div>
         <?php return ob_get_clean();
      }
   }
}
