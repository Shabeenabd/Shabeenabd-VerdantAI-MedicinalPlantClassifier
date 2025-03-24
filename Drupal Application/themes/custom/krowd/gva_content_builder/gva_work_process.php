<?php 
if(!class_exists('element_gva_work_process')):
   class element_gva_work_process{
      public function render_form(){
         $fields = array(
            'type'      => 'gsc_work_process',
            'title'  => t('Work Process'), 
            'size'      => 3, 
            
            'fields' => array(
               array(
                  'id'     => 'title',
                  'type'      => 'text',
                  'title'  => t('Title'),
                  'admin'     => true
               ),
               array(
                  'id'     => 'tabs',
                  'type'      => 'tabs',
                  'title'  => t('Work Process'),
                  'desc'      => t('You can use Drag & Drop to set the order.'),
               ),               
               
               array(
                  'id'     => 'el_class',
                  'type'      => 'text',
                  'title'  => t('Extra class name'),
                  'desc'      => t('Style particular content element differently - add a class name and refer to it in custom CSS.'),
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
            ),                                           
         );
      return $fields;
      }

      public static function render_content( $attr = array(), $content = '' ){
         extract(gavias_merge_atts(array(
            'title'           => '',
            'tabs'            => '',            
            'animate'         => '',
            'animate_delay'   => '',
            'el_class'        => ''
         ), $attr));
         $_id = 'workprocess-' . gavias_content_builder_makeid();
         $classes = $el_class;
         if($animate) $classes .= ' wow ' . $animate;
   
         ob_start();
         ?>
         <div class="gsc-workprocess <?php print $classes ?> " id="<?php print $_id; ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>
            <ul class="service-timeline post-area">
              <?php
               if( is_array( $tabs ) ){ 
                  $i=0;
                  foreach( $tabs as $tab ): $i++;
               ?>
               <li class="entry-timeline clearfix">
                  <div class="hentry skrollable skrollable-between" data-bottom-top="opacity: 0;" data-center-bottom="opacity: 1;" data-top-bottom="opacity: 0;">
                     <div class="icon"><span class="<?php print $tab['icon']; ?>"></span></div>    
                     <div class="hentry-box clearfix">
                        <div class="content-inner">
                           <div class="title"><?php print $tab['title'] ?></div>
                           <div class="content"><?php print $tab['content'] ?></div>
                        </div>   
                    </div>
                 </div> 
               </li>
               <?php endforeach;  ?>  
               <?php } ?>   
            </ul>
         </div>
         <?php return ob_get_clean() ?>
      <?php    
      }
 
   }

endif;