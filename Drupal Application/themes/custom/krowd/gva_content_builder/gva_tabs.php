<?php 
if(!class_exists('element_gva_tabs')){
   class element_gva_tabs{
      public function render_form(){
         $fields = array(
            'type'   => 'gsc_tabs',
            'title'  => t('Tabs'), 
            'fields' => array(      
               array(
                  'id'        => 'title',
                  'type'      => 'text',
                  'title'     => t('Title for admin'),
                  'admin'     => true
               ),
               array(
                  'id'        => 'type',
                  'type'      => 'select',
                  'options'   => array(
                     'horizontal'   => 'Horizontal',
                     'horizontal_icon'   => 'Horizontal Icon',
                     'vertical'     => 'Vertical', 
                  ),
                  'title'  => t('Style'),
                  'desc'      => t('Vertical tabs works only for column widths: 1/2, 3/4 & 1/1'),
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
               'id'        => "title_{$i}",
               'type'      => 'text',
               'title'     => t("Title {$i}")
            );
            $fields['fields'][] = array(
               'id'        => "title_content_{$i}",
               'type'      => 'text',
               'title'     => t("Title Content {$i}")
            );
            $fields['fields'][] = array(
               'id'           => "content_{$i}",
               'type'         => 'textarea',
               'title'        => t("Content {$i}")
            );
            $fields['fields'][] = array(
               'id'           => "image_{$i}",
               'type'         => 'upload',
               'title'        => t("Image {$i}")
            );
         }
         return $fields;
      }

      public static function render_content( $attr = array(), $content = '' ){
         global $base_url;
         $default = array(
            'title'           => '',
            'title_content'   => '',
            'uid'             => 'tab-',
            'type'            => '',
            'el_class'        => '',
            'animate'         => '',
            'animate_delay'   => ''
         );  

            
         for($i=1; $i<=8; $i++){
            $default["title_{$i}"] = '';
            $default["title_content_{$i}"] = '';
            $default["content_{$i}"] = '';
            $default["image_{$i}"] = '';
         }   
         extract(gavias_merge_atts($default, $attr)); 
         
         $_id = gavias_content_builder_makeid();
         $uid .= $_id;
         if($animate) $el_class .= ' wow ' . $animate;
         
         ob_start() ?>
         <div class="gsc-tabs <?php print $el_class ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>
            <div class="tabs_wrapper tabs_<?php print $type ?>">
               <ul class="nav nav-tabs">
                  <?php for($i=1; $i<=8; $i++){ ?>
                     <?php 
                        $title = "title_{$i}";
                        $title_content = "title_content_{$i}";
                        $content = "content_{$i}";
                        $image = "image_{$i}";
                     ?>
                     <?php if($$title){ ?>
                        <li><a <?php print($i==1?'class="active"':'') ?> data-toggle="tab" href="#<?php print ($uid .'-'. $i) ?>">  <?php print $$title ?> </a></li>
                     <?php } ?>
                  <?php } ?>
               </ul>
               <div class="tab-content">
                  <?php for($i=1; $i<=8; $i++){ ?>
                     <?php 
                        $title = "title_{$i}";
                        $title_content = "title_content_{$i}";
                        $content = "content_{$i}";
                        $image = "image_{$i}";
                     ?>
                     
                        <div id="<?php print($uid .'-'. $i) ?>" class="tab-pane fade show <?php print($i==1?'active':'') ?>">
                           <div class="box-content">
                              <div class="content-inner">
                                 <?php if($$title_content){ ?>
                                    <h2 class="title-content">
                                       <?php print $$title_content ?>
                                    </h2>
                                 <?php } ?> 
                                 <?php if($$content){ ?>
                                    <div class="content">
                                       <?php print $$content ?>
                                    </div>
                                 <?php } ?> 
                              </div> 
                              <?php if($$image){ ?>
                                 <div class="image"><img src="<?php echo ($base_url . $$image) ?>" /></div>
                              <?php } ?>                                             
                           </div>  
                        </div>                   
                  <?php } ?>
               </div>
            </div>
         </div>
         <?php return ob_get_clean();
      }
   }
}


